# mgcv SPT bleeding-on-probing model for the memory / runtime benchmark: reads the analysis sample, builds the model matrices and fits.
# Same model as benchmark_spt_torchpalmm.py: per baseline class x sex group one follow-up curve and one curve of BOP at the first visit, the same fixed effects, a random intercept with its own variance in each class x sex x smoker stratum + random slopes on age, teeth at the visit and log time since the preceding visit (diagonal D), iid residuals.
# usage: Rscript benchmark_spt_mgcv_more_re.R MODE CSV OUT.csv K     MODE = baseline (packages and data only) | bam | gamm
# Both fits use a diagonal D: bam via s(pat, by = grp_re, bs = 're') (one variance per stratum), gamm via nlme::pdDiag on the stratum indicators + the three slopes. Covariances between the stratum intercepts are not identified (a patient is in one stratum).
args <- commandArgs(trailingOnly = TRUE)
mode <- args[1]; csv <- args[2]; out_csv <- args[3]; k_arg <- args[4]   # k_arg: basis functions per curve, or "match" = as many as torchpalmm (a knot at every distinct value, per curve)
suppressPackageStartupMessages(library(mgcv))
d <- read.csv(csv)
d <- d[order(d$pat_id, d$spt_id), ]
d$pat <- factor(d$pat_id); d$grp <- factor(d$grp); d$grp_re <- factor(d$grp_re)
if (mode == "baseline") { cat(sprintf("RESULT mode=baseline rows=%d\n", nrow(d))); quit() }
x_cols <- c("age_z", "year_z", "teeth_z", "logdays_z", "cls2", "cls3", "female", "ever", "cls2_female", "cls3_female")
fixed <- paste("y ~", paste(x_cols, collapse = " + "))
lv <- levels(d$grp); kn <- NULL
lv_re <- levels(d$grp_re); slopes <- c("age_z", "teeth_z", "logdays_z")
gcols <- paste0("gi", seq_along(lv_re)); for (g in seq_along(lv_re)) d[[gcols[g]]] <- as.numeric(d$grp_re == lv_re[g])   # stratum indicators: the stratum-specific random intercept in gamm
curves <- expand.grid(g = seq_along(lv), v = c("fu", "bop_first"), stringsAsFactors = FALSE)[, c("v", "g")]
if (k_arg == "match") {   # torchpalmm's basis: per group a natural cubic spline with a knot at every distinct value of that group (k = distinct values, centred: k - 1 coefficients)
  kn <- list()
  for (g in seq_along(lv)) d[[paste0("o", g)]] <- factor(ifelse(d$grp == lv[g], "in", "out"), levels = c("out", "in"), ordered = TRUE)   # ordered by: a smooth for "in" only
  curves$var <- paste0(curves$v, "_g", curves$g)                                            # a covariate copy per group, so its knots can be the group's own distinct values
  for (i in seq_len(nrow(curves))) { d[[curves$var[i]]] <- d[[curves$v[i]]]; kn[[curves$var[i]]] <- sort(unique(d[[curves$v[i]]][d$grp == lv[curves$g[i]]])) }
  curves$label <- sprintf("s(%s):o%din", curves$var, curves$g)
  smooths <- paste(sprintf("s(%s, by = o%d, bs = 'cr', k = %d)", curves$var, curves$g, lengths(kn[curves$var])), collapse = " + ")
} else {
  curves$var <- curves$v; curves$label <- sprintf("s(%s):grp%s", curves$v, lv[curves$g])
  smooths <- paste(sprintf("s(%s, by = grp, bs = 'cr', k = %d)", c("fu", "bop_first"), as.integer(k_arg)), collapse = " + ")   # one centred curve per class x sex group and smooth
}
if (mode == "bam") {
  t <- system.time(m <- bam(as.formula(paste(fixed, "+", smooths, "+ s(pat, by = grp_re, bs = 're') + s(age_z, pat, bs = 're') + s(teeth_z, pat, bs = 're') + s(logdays_z, pat, bs = 're')")),
                            data = d, knots = kn, method = "fREML", discrete = FALSE, nthreads = 1))
  cf <- coef(m)
  sdre <- sqrt(m$sig2 / m$sp[c(paste0("s(pat):grp_re", lv_re), paste0("s(", slopes, ",pat)"))])   # random-effect SD = residual SD / sqrt(smoothing parameter); one intercept SD per stratum
  n_re <- length(lv_re)
  vcomp <- c(setNames(unname(sdre[seq_len(n_re)]), paste0("sd_intercept, ", lv_re)), sd_age = unname(sdre[n_re + 1]), sd_teeth = unname(sdre[n_re + 2]), sd_logdays = unname(sdre[n_re + 3]), sigma = sqrt(m$sig2))
} else {
  t <- system.time(m <- gamm(as.formula(paste(fixed, "+", smooths)), data = d, knots = kn,
                             random = list(pat = nlme::pdDiag(as.formula(paste("~ 0 +", paste(c(gcols, slopes), collapse = " + "))))), method = "REML"))
  cf <- coef(m$gam)
  vcomp <- tryCatch({                                                                        # nlme variance components of the patient level
    vc <- VarCorr(m$lme); i <- which(startsWith(rownames(vc), "pat ="))[1]
    n_re <- length(lv_re); sds <- as.numeric(vc[i + seq_len(n_re + 3), "StdDev"])
    c(setNames(sds[seq_len(n_re)], paste0("sd_intercept, ", lv_re)), sd_age = sds[n_re + 1], sd_teeth = sds[n_re + 2], sd_logdays = sds[n_re + 3], sigma = m$lme$sigma)
  }, error = function(e) c(setNames(rep(NA_real_, length(lv_re)), paste0("sd_intercept, ", lv_re)), sd_age = NA_real_, sd_teeth = NA_real_, sd_logdays = NA_real_, sigma = m$lme$sigma))
}
gm <- if (mode == "bam") m else m$gam
ts <- system.time(sm <- summary(gm))                                                         # SEs and EDFs (the inference step; torchpalmm's summary is timed the same way)
## torchpalmm's curve constant: the least-squares line of f over the group's distinct values is 0 at the all-rows mean of x (mgcv centres differently).
## Shifting each group's two curve constants into its intercept changes only the intercept and the class / sex dummies; estimates and Bayesian SEs (Vp) are mapped linearly.
b <- coef(gm); p <- length(b); grp_terms <- c("(Intercept)", "cls2", "cls3", "female", "cls2_female", "cls3_female")
Gm <- matrix(0, length(lv), p); M <- matrix(0, length(lv), length(grp_terms))
for (g in seq_along(lv)) { r0 <- d[which(d$grp == lv[g])[1], ]; M[g, ] <- c(1, unlist(r0[grp_terms[-1]])) }
for (i in seq_len(nrow(curves))) {
  g <- curves$g[i]; u <- sort(unique(d[[curves$v[i]]][d$grp == lv[g]]))
  nd <- d[rep(which(d$grp == lv[g])[1], length(u)), ]; nd[[curves$var[i]]] <- u
  s_ <- gm$smooth[[which(sapply(gm$smooth, `[[`, "label") == curves$label[i])]]; cols <- s_$first.para:s_$last.para
  A <- cbind(1, u); cvec <- c(1, mean(d[[curves$v[i]]])) %*% solve(crossprod(A), t(A))
  Gm[g, cols] <- Gm[g, cols] + cvec %*% predict(gm, nd, type = "lpmatrix")[, cols, drop = FALSE]
}
Tm <- diag(p); idx <- match(grp_terms, names(b)); Tm[idx, ] <- Tm[idx, ] + solve(M) %*% Gm
b2 <- setNames(drop(Tm %*% b), names(b)); se2 <- setNames(sqrt(diag(Tm %*% gm$Vp %*% t(Tm))), names(b))
write.csv(data.frame(term = c(x_cols, names(vcomp)), estimate = c(unname(b2[x_cols]), unname(vcomp)), se = c(unname(se2[x_cols]), rep(NA_real_, length(vcomp)))), out_csv, row.names = FALSE)
var_lab <- c(fu = "Follow-up time", bop_first = "BOP at first visit")
write.csv(data.frame(term = paste0(var_lab[curves$v], ", ", lv[curves$g]), edf = sm$s.table[curves$label, "edf"]), sub("\\.csv$", "_edf.csv", out_csv), row.names = FALSE)
smooth_basis <- sum(sapply(gm$smooth, function(sm_) if (sm_$label %in% curves$label) sm_$last.para - sm_$first.para + 1 else 0))   # curve coefficients (linear + penalized), comparable with torchpalmm's smooth_coefs
converged <- if (mode == "bam") isTRUE(m$converged) else TRUE                                 # gamm: nlme returned a fit (no separate flag)
cat(sprintf("RESULT mode=%s rows=%d subjects=%d total_basis=%d smooth_basis=%d converged=%d fit_seconds=%.1f summary_seconds=%.1f\n", mode, nrow(d), nlevels(d$pat), length(cf), smooth_basis, as.integer(converged), t[["elapsed"]], ts[["elapsed"]]))
