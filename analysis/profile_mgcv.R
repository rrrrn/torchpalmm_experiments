
args <- commandArgs(trailingOnly = TRUE); mode <- args[1]
suppressPackageStartupMessages(library(mgcv))
d <- read.csv("../processed/analytic_sample_T1.csv")
d <- d[order(d$Subject, d$wear_day), ]
if (is.character(d$Gender)) d$Gender <- as.numeric(d$Gender == "F")                                  # female = 1, as in the notebook
d$Subject <- factor(d$Subject); d$grp <- factor(d$Gender * 2 + d$weekend_night)                # sex x weekday / weekend night
d$first <- !duplicated(d$Subject)                                                                # AR(1) restarts at each subject's first night
fixed <- "SleepDur_FINAL ~ scale(Age) + Gender * weekend_night + WAZ + HAZ + scale(bps)"
smooths <- "s(ALR1_00_19, by = grp, bs = 'cr', k = 20) + s(ALR1_19_24, by = grp, bs = 'cr', k = 20) + s(ALR2_00_19, by = grp, bs = 'cr', k = 20) + s(ALR2_19_24, by = grp, bs = 'cr', k = 20)"
if (mode == "bam") {
  t <- system.time(m <- bam(as.formula(paste(fixed, "+", smooths, "+ s(Subject, bs = 're')")), data = d, method = "fREML",
                            discrete = TRUE, nthreads = 1, rho = 0.157, AR.start = d$first))
} else if (mode == "gamm") {
  t <- system.time(m <- gamm(as.formula(paste(fixed, "+", smooths)), data = d, random = list(Subject = ~1),
                             correlation = corAR1(form = ~ wear_day | Subject), method = "REML"))
} else t <- system.time(NULL)
cat(sprintf("RESULT mode=%s rows=%d subjects=%d fit_seconds=%.1f\n", mode, nrow(d), nlevels(d$Subject), t[["elapsed"]]))
