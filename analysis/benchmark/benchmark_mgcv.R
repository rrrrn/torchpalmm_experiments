# mgcv cell-means model for the memory / runtime benchmark: reads the analysis sample, builds the model matrices and fits.
# usage: Rscript benchmark_mgcv.R MODE CSV OUT.csv K RHO [SUMMARY]     MODE = baseline (packages and data only) | bam | gamm; SUMMARY = 1 also runs summary() (SEs, p-values, curve tests)
args <- commandArgs(trailingOnly = TRUE)
mode <- args[1]; csv <- args[2]; out_csv <- args[3]; k <- as.integer(args[4]); rho <- as.numeric(args[5]); do_summary <- length(args) >= 6 && args[6] == "1"
suppressPackageStartupMessages(library(mgcv))
d <- read.csv(csv)
d <- d[order(d$Subject, d$wear_day), ]
if (is.character(d$Gender)) d$Gender <- as.numeric(d$Gender == "F")                      # female = 1
d$Subject <- factor(d$Subject)
d$grp <- factor(d$Gender * 2 + d$weekend_night)                                          # sex x weekday / weekend night
d$first <- !duplicated(d$Subject)                                                        # AR(1) restarts at each subject's first night
if (mode == "baseline") { cat(sprintf("RESULT mode=baseline rows=%d\n", nrow(d))); quit() }
covs <- c("ALR1_00_19", "ALR1_19_24", "ALR2_00_19", "ALR2_19_24")
fixed <- "SleepDur_FINAL ~ Age_scaled + Gender * weekend_night + WAZ + HAZ + bps_scaled"
smooths <- paste(sprintf("s(%s, by = grp, bs = 'cr', k = %d)", covs, k), collapse = " + ")   # one centred curve per group and exposure x interval
if (mode == "bam") {
  t <- system.time(m <- bam(as.formula(paste(fixed, "+", smooths, "+ s(Subject, bs = 're')")), data = d, method = "fREML",
                            discrete = TRUE, nthreads = 1, rho = rho, AR.start = d$first))
  cf <- coef(m)
} else {
  t <- system.time(m <- gamm(as.formula(paste(fixed, "+", smooths)), data = d, random = list(Subject = ~1),
                             correlation = corAR1(form = ~ wear_day | Subject), method = "REML"))
  cf <- coef(m$gam)
}
sum_secs <- 0
if (do_summary) sum_secs <- system.time(print(summary(if (mode == "bam") m else m$gam)))[["elapsed"]]
keep <- c("Age_scaled", "Gender", "WAZ", "HAZ", "bps_scaled", "weekend_night", "Gender:weekend_night")
write.csv(data.frame(term = keep, estimate = unname(cf[keep])), out_csv, row.names = FALSE)
cat(sprintf("RESULT mode=%s rows=%d subjects=%d total_basis=%d fit_seconds=%.1f summary_seconds=%.1f\n", mode, nrow(d), nlevels(d$Subject), length(cf), t[["elapsed"]], sum_secs))
