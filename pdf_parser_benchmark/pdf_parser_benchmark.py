# -*- coding: utf-8 -*-
"""
Dependencies:

    * pymupdf4llm
    * docling
    * magic-pdf
    * marker
    * markitdown
    * mistralai
"""
import base64
import os

from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, EasyOcrOptions
from docling.document_converter import DocumentConverter, FormatOption, PdfFormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling_core.types.doc import ImageRefMode, PictureItem
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from markitdown import MarkItDown
from mistralai import Mistral
import pymupdf4llm
from tqdm import tqdm

os.makedirs('output', exist_ok=True)


def get_output_filename(file: str | os.PathLike, package: str) -> str:
    return f'output/{os.path.basename(file).removesuffix(".pdf")}_{package}.md'


def pymupdf_parser(file: str | os.PathLike):
    with open(get_output_filename(file, 'pymupdf'), 'wb') as f:
        f.write(pymupdf4llm.to_markdown(file, write_images=True, image_path='output').encode())
    return


def docling_parser(file: str | os.PathLike):
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = 2  # 1 = 72 dpi, 2 = 144 dpi
    pipeline_options.generate_picture_images = True
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    result = converter.convert(file)

    # Images
    pdf_filename = result.input.file.stem
    img_no = 0
    for element, _level in result.document.iterate_items():
        if isinstance(element, PictureItem):
            img_no += 1
            img_filename = f'output/{pdf_filename}-{img_no}_docling.png'
            with open(img_filename, 'wb') as f:
                element.get_image(result.document).save(f, 'png')

    # Text
    result.document.save_as_markdown(get_output_filename(file, 'docling'),
                                     image_mode=ImageRefMode.REFERENCED)
    return


def mineru_parser(file: str | os.PathLike):
    reader1 = FileBasedDataReader('')
    pdf_bytes = reader1.read(file)
    ds = PymuDocDataset(pdf_bytes)

    image_writer, md_writer = FileBasedDataWriter('output'), FileBasedDataWriter('output')
    if ds.classify() == SupportedPdfParseMethod.OCR:
        infer_result = ds.apply(doc_analyze, ocr=True)
        pipe_result = infer_result.pipe_ocr_mode(image_writer)
    else:
        infer_result = ds.apply(doc_analyze, ocr=False)
        pipe_result = infer_result.pipe_txt_mode(image_writer)

    pipe_result.dump_md(md_writer, get_output_filename(file, 'mineru'), 'output')
    return


def markitdown_parser(file: str | os.PathLike):
    md = MarkItDown()
    result = md.convert(file)
    with open(get_output_filename(file, 'markitdown'), 'w', encoding='utf-8') as f:
        f.write(result.text_content)
    return


def marker_parser(file: str | os.PathLike):
    converter = PdfConverter(artifact_dict=create_model_dict(device='cuda'))
    rendered = converter(file)
    text, _, images = text_from_rendered(rendered)
    with open(get_output_filename(file, 'marker'), 'w', encoding='utf-8') as f:
        f.write(text)
    for img_ref, img in images.items():
        img.save(f'output/{img_ref}')
    return


def mistral_parser(file: str | os.PathLike):
    client = Mistral(os.environ['MISTRAL_API_KEY'])
    uploaded_pdf = client.files.upload(
        file={
            'file_name': 'file',
            'content': open(file, "rb"),
        },
        purpose='ocr'
    )
    signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)
    ocr_response = client.ocr.process(
        model='mistral-ocr-latest',
        document={
            'type': 'document_url',
            'document_url': signed_url.url
        },
        include_image_base64=True
    )
    with open(get_output_filename(file, 'mistral'), 'w', encoding='utf-8') as f:
        f.write(ocr_response.pages[0].markdown)
    for img in ocr_response.pages[0].images:
        with open(f'output/{img.id}', 'wb') as img_file:
            img_file.write(
                base64.b64decode(img.image_base64.removeprefix('data:image/jpeg;base64,'))
            )
    client.files.delete(file_id=uploaded_pdf.id)
    return


def ilyarice_parser(file: str | os.PathLike):
    """From https://github.com/IlyaRice/RAG-Challenge-2/blob/main/src/pdf_parsing.py"""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    ocr_options = EasyOcrOptions(lang=['en'], force_full_page_ocr=False)
    pipeline_options.ocr_options = ocr_options
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    pipeline_options.images_scale = 2
    pipeline_options.generate_picture_images = True
    accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CUDA)
    pipeline_options.accelerator_options = accelerator_options
    format_options = {
        InputFormat.PDF: FormatOption(
            pipeline_cls=StandardPdfPipeline,
            pipeline_options=pipeline_options,
            backend=DoclingParseV2DocumentBackend
        )
    }

    result = DocumentConverter(format_options=format_options).convert(file)
    pdf_filename = result.input.file.stem
    img_no = 0
    for element, _level in result.document.iterate_items():
        if isinstance(element, PictureItem):
            img_no += 1
            img_filename = f'output/{pdf_filename}-{img_no}_ilyarice.png'
            with open(img_filename, 'wb') as f:
                element.get_image(result.document).save(f, 'png')
    result.document.save_as_markdown(get_output_filename(file, 'ilyarice'),
                                     image_mode=ImageRefMode.REFERENCED)
    return


def main():
    pdfs = [f'pdf/{i}' for i in os.listdir('pdf')]
    parsers = {
        'pymupdf': pymupdf_parser,
        # 'docling': docling_parser,
        # 'mineru': mineru_parser,
        # 'markitdown': markitdown_parser,
        # 'marker': marker_parser,
        # 'mistral': mistral_parser,
        # 'ilyarice': ilyarice_parser
    }
    # pymupdf4llm and mineru requires conflicting versions of pymupdf, so run them separately
    for pdf in tqdm(pdfs):
        for name, parser in parsers.items():
            parser(pdf)
    return


if __name__ == '__main__':
    main()
