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


/* Benchmark ******************************************************************/
// California smoking data
cap frame drop timing
mkf timing
frame timing {
    set obs 1000    // run 1,000 times
    gen stata = .
}
_dots 0, reps(100)
forvalues i = 1/1000 {
    timer clear
    timer on 1
    cap synth cigsale $covariates, trunit(3) trperiod(1989) ///
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


/* For post 2 *****************************************************************/
// Get loss and weights so can compare with Python
clear all
use "https://github.com/microsoft/SparseSC/raw/master/replication/smoking.dta"
xtset state year
synth cigsale beer(1984(1)1988) lnincome retprice age15to24 cigsale(1988) ///
      cigsale(1980) cigsale(1975), trunit(3) trperiod(1989) nested allopt
clear
mat v = vecdiag(e(V_matrix))
svmat v, names(v)
xpose, clear
svmat e(W_weights), names(w)
drop w1
ren v1 v
ren w2 w
export delimited "data/stata_synth_nested.csv", delimit(",") replace

// Time
forvalues i = 1/10 {
    timeit 1: synth cigsale beer(1984(1)1988) lnincome retprice age15to24 ///
              cigsale(1988) cigsale(1980) cigsale(1975), tru(3) trp(1989)
}
timer list
local time = r(t1) / 10
clear all
insobs 1
gen time = `time'
export delimited "data/stata_time_nested.csv", novar replace