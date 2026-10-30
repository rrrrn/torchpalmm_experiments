## Table 1 for the SPT bleeding-on-probing analysis: patient and visit characteristics by sex
## Same layout as table1_body.tex (kableExtra, booktabs, grouped rows). Run from analysis/:
##   Rscript table1_spt.R
## Analysis sample as in fit.ipynb: visits 2 onward with a non-missing bleeding on probing outcome.
## Writes table1_spt.tex
## Data: Supportive Periodontal Therapy Clinical Examination Data (PhysioNet), https://physionet.org/content/supportive-periodontal-therapy/1.0.0/
##   Download it into analysis/spt_data/ (not tracked by git), then run curate_spt_bleeding.py to build the curated CSV.
suppressPackageStartupMessages({
  library(dplyr)
  library(kableExtra)
})

v <- read.csv("spt_data/curated_spt_bleeding.csv") |>
  filter(spt_id >= 2, !is.na(bleeding_pct))                  # first SPT visit excluded; missing outcome dropped

## one row per patient: baseline characteristics (initial examination) and follow-up
p <- v |>
  group_by(pat_id) |>
  summarise(sex = first(sex), age = first(age_baseline), ever = first(smoking_status != "non-smoker"),
            cls = first(baseline_diagnosis), teeth = first(n_teeth_baseline), pct5 = first(pct_from5mm_baseline),
            fu = max(cum_followup_years), nvis = n(), .groups = "drop")

fmt <- function(x, d = 1) formatC(x, format = "f", digits = d, big.mark = ",")
mean_sd <- function(x) sprintf("%s (%s)", fmt(mean(x)), fmt(sd(x)))
med_iqr <- function(x, d = 1) sprintf("%s [%s, %s]", fmt(median(x), d), fmt(quantile(x, .25), d), fmt(quantile(x, .75), d))
n_pct <- function(x) sprintf("%s (%s)", fmt(sum(x), 0), fmt(100 * mean(x)))

groups <- list(Male = "male", Female = "female", Total = c("male", "female"))
column <- function(g) {
  pg <- p[p$sex %in% g, ]; vg <- v[v$sex %in% g, ]
  c(mean_sd(pg$age), n_pct(pg$ever), n_pct(pg$cls == 1), n_pct(pg$cls == 2), n_pct(pg$cls == 3),
    mean_sd(pg$teeth), mean_sd(pg$pct5),
    med_iqr(pg$fu), med_iqr(pg$nvis, 0),
    fmt(nrow(vg), 0), mean_sd(vg$bleeding_pct), mean_sd(vg$age_at_spt), mean_sd(vg$n_teeth_spt),
    med_iqr(vg$days_since_prev_visit, 0), sprintf("%d--%d", min(vg$calendar_year), max(vg$calendar_year)))
}

labels <- c("Age (years)", "Ever-smoker, n (\\%)",
            "Disease severity class 1, n (\\%)", "Disease severity class 2, n (\\%)", "Disease severity class 3, n (\\%)",
            "Teeth (count)", "Sites with PPD $\\geq$ 5 mm (\\%)",
            "Follow-up duration (years), median [IQR]", "SPT visits per patient, median [IQR]",
            "Visits (count)", "Bleeding on probing (\\% of sites)", "Age at visit (years)", "Teeth at visit (count)",
            "Time since preceding visit (days), median [IQR]", "Calendar year, range")
tab <- data.frame(Characteristic = labels, sapply(groups, column), check.names = FALSE)
n_pat <- sapply(groups, function(g) sum(p$sex %in% g))
colnames(tab) <- c("Characteristic", sprintf("%s (N = %s)", names(groups), fmt(n_pat, 0)))

tex <- kbl(tab, format = "latex", booktabs = TRUE, escape = FALSE, align = c("l", rep("r", length(groups))), row.names = FALSE) |>
  kable_styling(latex_options = "scale_down") |>
  pack_rows("Baseline (initial examination)", 1, 7) |>
  pack_rows("Follow-up", 8, 9) |>
  pack_rows("SPT visits", 10, 15)
writeLines(as.character(tex), "table1_spt.tex")
print(tab, row.names = FALSE)
