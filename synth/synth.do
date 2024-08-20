********************************************************************************
* SCM in Stata                                                                 *
********************************************************************************
/* Hainmueller et al. (JASA 2010)'s official implementation *******************/
// Default
clear all
use "https://github.com/microsoft/SparseSC/raw/master/replication/smoking.dta"
xtset state year
global covariates = "cigsale(1970) cigsale(1971) cigsale(1972) cigsale(1973)"
global covariates = "$covariates cigsale(1974) cigsale(1975) cigsale(1976)"
global covariates = "$covariates cigsale(1977) cigsale(1978) cigsale(1979)"
global covariates = "$covariates cigsale(1980) cigsale(1981) cigsale(1982)"
global covariates = "$covariates cigsale(1983) cigsale(1984) cigsale(1985)"
global covariates = "$covariates cigsale(1986) cigsale(1987) cigsale(1988)"
synth cigsale $covariates, trunit(3) trperiod(1989) ///
    customV(1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1) ///
    keep("data/stata_synth.dta") replace
preserve
use "data/stata_synth.dta", clear    // save in .csv, so git is happy
compress
export delimited "data/stata_synth.csv", delimit(",") replace
erase "data/stata_synth.dta"
restore

// Nested optimisation
preserve
synth cigsale $covariates, trunit(3) trperiod(1989) nested ///
    customV(1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1) ///
    keep("data/stata_synth.dta") replace
use "data/stata_synth.dta", clear    // save in .csv, so git is happy
compress
export delimited "data/stata_synth_nested.csv", delimit(",") replace
erase "data/stata_synth.dta"
restore


/* Benchmark ******************************************************************/
// California smoking data
cap frame drop timing
mkf timing
frame timing {
    set obs 1000    // run 1,000 times
    gen stata = .
}
_dots 0, reps(100)
forvalues i = 1/700 {
    timer clear
    timer on 1
    cap synth cigsale $covariates, trunit(3) trperiod(1989) nested ///
        customV(1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1) ///
        keep("data/stata_synth.dta") replace
    if (_rc == 0) {
        timer off 1
        qui: timer list 1
        qui: frame timing: replace stata = r(t1) in `i'
        _dots `i' 0
    }
}
cap erase "data/stata_synth.dta"
frame timing {
    compress
    export delimited "data/stata_timing.csv", delimit(",") replace
}
clear all
