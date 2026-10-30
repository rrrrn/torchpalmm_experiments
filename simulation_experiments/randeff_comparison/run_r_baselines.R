# R baselines for the random-effects scaling experiment (sn=1, pz swept).
#   gamm : single smooth s(t) + fixed effects + random effects (pdDiag/pdSymm)
#   bam  : single smooth + bs="re" random effects (diagonal D only)
#
# Run from project root:
#   Rscript experiments/randeff_comparison/run_r_baselines.R --N 500 --pz 4 --D_type diagonal
#
# --only gamm|bam restricts the run to one method, writing to its own suffixed
# CSV (e.g. r_randeff_results_bam.csv) so gamm and bam can be submitted as
# separate, parallel SLURM jobs without racing on the same output file. The
# skip-set is still seeded from every results/r_randeff*.csv on disk, so a
# --only run never re-fits a combo an earlier combined (non-split) run already
# produced.

local({
  lib_dir <- path.expand("~/R/library")
  dir.create(lib_dir, recursive = TRUE, showWarnings = FALSE)
  .libPaths(c(lib_dir, .libPaths()))
  pkgs <- c("mgcv", "jsonlite")
  miss <- pkgs[!pkgs %in% installed.packages()[, "Package"]]
  if (length(miss)) {
    on_mac <- Sys.info()[["sysname"]] == "Darwin"
    install.packages(miss, repos = "https://cran.r-project.org", quiet = TRUE,
                     lib = if (on_mac) NULL else lib_dir,
                     type = if (on_mac) "binary" else getOption("pkgType"))
  }
})
suppressPackageStartupMessages({
  library(mgcv)
  library(jsonlite)
  library(nlme)
})

args0    <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args0, value = TRUE)
if (length(file_arg) > 0)
  setwd(dirname(normalizePath(sub("^--file=", "", file_arg))))

DATA_DIR <- Sys.getenv("RE_DATA_DIR", unset = "data")
TIME_CAP <- 1800


get_rss_mb <- function() {
  tryCatch({
    lines <- readLines("/proc/self/status")
    vmrss <- grep("^VmRSS:", lines, value = TRUE)
    if (length(vmrss) > 0)
      return(as.numeric(strsplit(trimws(vmrss[1]), "\\s+")[[1]][2]) / 1024)
  }, error = function(e) NULL)
  tryCatch({
    res <- system2("ps", args = c("-o", "rss=", "-p", Sys.getpid()), stdout = TRUE)
    as.numeric(trimws(res[1])) / 1024
  }, error = function(e) NA_real_)
}

# Set TRUE iff the most recent with_time_cap() call hit the elapsed-time cap
# (as opposed to a non-timeout R error). Read right after each call.
.fit_timed_out <- FALSE

with_time_cap <- function(expr, cap = TIME_CAP) {
  setTimeLimit(elapsed = cap, transient = TRUE)
  on.exit(setTimeLimit(elapsed = Inf), add = TRUE)
  .fit_timed_out <<- FALSE
  tryCatch(expr, error = function(e) {
    if (grepl("elapsed time limit", e$message, fixed = TRUE)) {
      .fit_timed_out <<- TRUE
      message(sprintf("TIMEOUT: exceeded %ds", cap))
    } else {
      message("ERROR: ", e$message)
    }
    NULL
  })
}

# A config (N, pz, D_type, corr) that racks up too many timeouts on ANY method
# is almost certainly intractable for the rest of its repeats too -- rather
# than burn the full TIME_CAP on each remaining rid, abort the config outright
# and record it so no future invocation (this job, a resubmission, or the
# sibling --only job fitting the other method) re-attempts it. Consumed by
# analyze_results.py's _load_deferred().
DEFER_THRESHOLD <- 10
DEFERRED_CSV    <- "results/r_deferred.csv"

# method-specific: a deferral recorded for "bam" must not silence a separate
# --only gamm job still making progress on the same (N, pz, D_type, corr),
# and vice versa -- that pairing is exactly what run_array_randeff.sh's
# separate gamm/bam array jobs are for.
is_config_deferred <- function(N, pz, D_type, corr, method) {
  if (!file.exists(DEFERRED_CSV)) return(FALSE)
  d <- tryCatch(read.csv(DEFERRED_CSV), error = function(e) NULL)
  if (is.null(d) || nrow(d) == 0) return(FALSE)
  any(d$N == N & d$pz == pz & d$D_type == D_type & d$corr == corr & d$method == method)
}

defer_config <- function(N, pz, D_type, corr, method, n_timeouts) {
  dir.create("results", showWarnings = FALSE)
  row <- data.frame(N = N, pz = pz, D_type = D_type, corr = corr,
                     method = method, n_timeouts = n_timeouts,
                     stringsAsFactors = FALSE)
  write.table(row, DEFERRED_CSV, sep = ",",
              col.names = !file.exists(DEFERRED_CSV),
              append = file.exists(DEFERRED_CSV),
              row.names = FALSE, quote = TRUE)
  cat(sprintf(paste("DEFERRED: N=%d pz=%d D=%s C=%s -- %s exceeded %d timeouts,",
                     "aborting remaining reps for this configuration\n"),
              N, pz, D_type, corr, method, n_timeouts))
}

trapz <- function(x, y) sum(diff(x) * (head(y, -1) + tail(y, -1)) / 2)

## True curve f0 (mirrors MIXED_SS_CURVES[0] in SimScripts/LMMSS.py)
true_curve <- function(t) 2 * sin(pi * t)

compute_ise <- function(t_grid, fitted, true) {
  fc <- fitted - mean(fitted)
  tc <- true   - mean(true)
  trapz(t_grid, (fc - tc)^2)
}

# =============================================================================
# RANDEFF (sn = 1, pz swept)
# =============================================================================

run_randeff <- function(cfg, N_list, pz_list, D_type_list, corr_list, only = NULL) {
  slice_mode <- length(N_list) == 1 && length(pz_list) == 1 && length(D_type_list) == 1 && length(corr_list) == 1
  suffix     <- if (!is.null(only)) sprintf("_%s", only) else ""
  out_csv    <- if (slice_mode)
    sprintf("results/r_randeff_N%d_pz%d_D%s_C%s%s.csv", N_list, pz_list, D_type_list, corr_list, suffix)
  else
    sprintf("results/r_randeff_results%s.csv", suffix)

  dir.create("results", showWarnings = FALSE)
  done <- list(); write_header <- TRUE
  if (file.exists(out_csv)) write_header <- FALSE

  # Skip-set is seeded from EVERY existing r_randeff*.csv, not just out_csv --
  # gamm and bam used to share one combined file fit in one job; now that they
  # can run as separate --only jobs writing their own suffixed file, a fresh
  # --only run must still see (and skip) rows already produced by that old
  # combined file, or by a previous non-split run, instead of re-fitting them.
  existing_csvs <- Sys.glob("results/r_randeff*.csv")
  if (length(existing_csvs) > 0) {
    ex_all <- do.call(rbind, lapply(existing_csvs, function(f)
      tryCatch(read.csv(f), error = function(e) NULL)))
    if (!is.null(ex_all) && nrow(ex_all) > 0) {
      keys <- paste(ex_all$N, ex_all$pz, ex_all$D_type, ex_all$corr,
                    ex_all$repeat_id, ex_all$method, sep = "|")
      done <- as.list(unique(keys))
      cat(sprintf("Resuming: %d rows already done across %d existing file(s)\n",
                  length(done), length(existing_csvs)))
    }
  }

  append_row <- function(row_df) {
    # write.table appends by position, and the row templates below list
    # n_iter/mem_mb in different orders: match the file's header by name.
    if (!write_header)
      row_df <- row_df[, names(read.csv(out_csv, nrows = 1, check.names = FALSE))]
    write.table(row_df, out_csv, sep = ",", col.names = write_header,
                append = !write_header, row.names = FALSE, quote = TRUE)
    write_header <<- FALSE
  }

  applicable_methods <- function(D_type, corr)
    if (D_type == "diagonal" && corr == "iid") c("gamm", "bam") else "gamm"

  px     <- cfg$px
  t_grid <- seq(0, 1, length.out = 500)
  g_true <- true_curve(t_grid)

  for (N in N_list) {
    for (pz in pz_list) {
      for (D_type in D_type_list) {
        for (corr in corr_list) {

          # Methods THIS job actually attempts here (applicable_methods()
          # intersected with --only). Skip the config only once every one of
          # them is deferred -- a bam deferral must not silence a still-
          # productive separate --only gamm job on the same config.
          methods_here <- applicable_methods(D_type, corr)
          if (!is.null(only)) methods_here <- intersect(methods_here, only)
          config_deferred <- function() {
            length(methods_here) > 0 && all(vapply(
              methods_here, function(m) is_config_deferred(N, pz, D_type, corr, m),
              logical(1)))
          }

          if (config_deferred()) {
            cat(sprintf(paste("SKIP (deferred): N=%d pz=%d D=%s C=%s --",
                               "previously exceeded %d timeouts\n"),
                        N, pz, D_type, corr, DEFER_THRESHOLD))
            next
          }
          gamm_timeouts <- 0L; bam_timeouts <- 0L
          gamm_successes <- 0L; bam_successes <- 0L

          for (rid in cfg$repeat_ids) {
            # Re-check each rid too, not just on config entry: the sibling
            # --only job fitting the other method may defer this same config
            # mid-run, and there is no cheaper point to notice than here.
            if (config_deferred()) break
            gamm_attempted <- FALSE; gamm_timeout <- FALSE; gamm_converged <- FALSE
            bam_attempted  <- FALSE; bam_timeout  <- FALSE; bam_converged  <- FALSE
            path <- sprintf("%s/randeff_N%d_pz%d_D%s_C%s_repeat%d.json",
                            DATA_DIR, N, pz, D_type, corr, rid)
          d <- tryCatch(jsonlite::fromJSON(path, simplifyVector = TRUE),
                        error = function(e) { message("Missing: ", path); NULL })
          if (is.null(d)) next

          y   <- as.numeric(d$y)
          sv  <- matrix(as.numeric(unlist(d$splines_vars)), ncol = 1)
          X   <- matrix(as.numeric(unlist(d$X)), ncol = px)
          Z   <- matrix(as.numeric(unlist(d$Z)), ncol = pz)
          ids <- factor(as.integer(d$subject_id) + 1L)

          t_df <- as.data.frame(sv); colnames(t_df) <- "t1"
          X_df <- as.data.frame(X);  colnames(X_df) <- paste0("X", seq_len(px))
          Z_df <- as.data.frame(Z);  colnames(Z_df) <- paste0("Zs", seq_len(pz))
          df   <- cbind(data.frame(y = y, id = ids), t_df, X_df, Z_df)

          x_terms <- paste(paste0("X", seq_len(px)), collapse = " + ")
          k1      <- min(length(unique(t_df$t1)), 500)
          smooth_terms <- sprintf("s(t1, bs='cr', k=%d)", k1)

          true_D_diag <- diag(matrix(as.numeric(unlist(d$D)), nrow = pz))
          true_beta   <- as.numeric(d$beta)
          # Mirror the Python loader's fallback when rho isn't persisted.
          true_rho    <- if (!is.null(d$rho)) as.numeric(d$rho)
                         else if (corr == "compound_symmetry") 0.4 else 0.0

          ## ---- gamm --------------------------------------------------------
          key_gamm <- paste(N, pz, D_type, corr, rid, "gamm", sep = "|")
          if ((is.null(only) || only == "gamm") && !(key_gamm %in% done)) {
            gamm_attempted <- TRUE
            cat(sprintf("randeff N=%d pz=%d D=%s C=%s repeat=%d gamm ...\n",
                        N, pz, D_type, corr, rid))
            fixed_form <- as.formula(paste("y ~", smooth_terms, "+", x_terms))
            rand_form  <- as.formula(paste("~ 0 +",
                                            paste(paste0("Zs", seq_len(pz)),
                                                  collapse = "+")))
            pd_struct  <- if (D_type == "diagonal") pdDiag(rand_form) else pdSymm(rand_form)
            # Add residual correlation structure if CS
            corr_struct <- if (corr == "compound_symmetry") corCompSymm(form = ~ 1) else NULL

            mem_before <- get_rss_mb()
            t0  <- proc.time()["elapsed"]
            fit <- with_time_cap(
              if (is.null(corr_struct)) {
                gamm(fixed_form, random = list(id = pd_struct),
                     data = df, method = "REML",
                     control = lmeControl(opt = "optim",
                                          maxIter = 200, msMaxIter = 200))
              } else {
                gamm(fixed_form, random = list(id = pd_struct),
                     correlation = corr_struct,
                     data = df, method = "REML",
                     control = lmeControl(opt = "optim",
                                          maxIter = 200, msMaxIter = 200))
              }
            )
            t1     <- proc.time()["elapsed"]
            mem_mb <- get_rss_mb() - mem_before
            gamm_timeout <- isTRUE(.fit_timed_out)

            if (is.null(fit)) {
              append_row(data.frame(N=N, pz=pz, D_type=D_type, corr=corr, repeat_id=rid,
                                    method="gamm", converged=FALSE, timed_out=gamm_timeout,
                                    time_s=NA, n_iter=NA,
                                    mem_mb=NA, beta_mse=NA, beta_mae=NA,
                                    sigma2_abs_err=NA, D_mse=NA, D_mae=NA,
                                    D_full_mse=NA, rho_est=NA, rho_abs_err=NA,
                                    ise_macro=NA))
            } else {
              newdf <- data.frame(t1 = t_grid)
              for (j in seq_len(px)) newdf[[paste0("X", j)]] <- 0
              terms <- predict(fit$gam, newdata = newdf, type = "terms")
              col_idx <- grep("^s\\(t1\\)$", colnames(terms))
              smooth_hat <- if (length(col_idx) > 0) as.numeric(terms[, col_idx])
                            else rep(0, length(t_grid))
              ise <- compute_ise(t_grid, smooth_hat, g_true)

              converged  <- is.matrix(fit$lme$apVar)
              gamm_converged <- converged
              vc <- tryCatch(VarCorr(fit$lme), error = function(e) NULL)
              if (is.null(vc)) {
                append_row(data.frame(N=N, pz=pz, D_type=D_type, corr=corr, repeat_id=rid,
                                      method="gamm", converged=FALSE, timed_out=FALSE,
                                      time_s=as.numeric(t1-t0), n_iter=NA,
                                      mem_mb=mem_mb, beta_mse=NA, beta_mae=NA,
                                      sigma2_abs_err=NA, D_mse=NA, D_mae=NA,
                                      D_full_mse=NA, rho_est=NA, rho_abs_err=NA,
                                      ise_macro=ise))
              } else {
              D_diag_hat <- as.numeric(vc[paste0("Zs", seq_len(pz)), "StdDev"])^2
              sigma2_hat <- as.numeric(vc["Residual", "StdDev"])^2
              beta_hat   <- coef(fit$gam)[paste0("X", seq_len(px))]

              # Estimated residual correlation. corCompSymm has a single ρ; with
              # no correlation structure (iid) the model assumes ρ = 0.
              rho_hat <- tryCatch({
                cs <- fit$lme$modelStruct$corStruct
                if (!is.null(cs)) as.numeric(coef(cs, unconstrained = FALSE))[1]
                else 0.0
              }, error = function(e) NA_real_)

              # Full D matrix recovery (for unstructured D)
              D_full_mse <- NA_real_
              if (D_type == "unstructured") {
                tryCatch({
                  D_hat_full <- getVarCov(fit$lme, individual = 1, type = "random.effects")[[1]]
                  D_hat_full <- matrix(D_hat_full, nrow = pz)
                  true_D_full <- matrix(as.numeric(unlist(d$D)), nrow = pz)
                  if (all(dim(D_hat_full) == dim(true_D_full))) {
                    D_full_mse <- mean((D_hat_full - true_D_full)^2)
                  }
                }, error = function(e) NULL)
              } else {
                # For diagonal, D_full_mse = D_mse (only diagonal elements)
                true_D_full <- diag(true_D_diag)
                D_hat_full <- diag(D_diag_hat)
                D_full_mse <- mean((D_hat_full - true_D_full)^2)
              }

              append_row(data.frame(
                N=N, pz=pz, D_type=D_type, corr=corr, repeat_id=rid, method="gamm",
                converged=converged, timed_out=FALSE,
                time_s=as.numeric(t1-t0), mem_mb=mem_mb,
                n_iter=if (!is.null(fit$lme$numIter)) as.integer(fit$lme$numIter) else NA_integer_,
                beta_mse=mean((beta_hat - true_beta)^2),
                beta_mae=mean(abs(beta_hat - true_beta)),
                sigma2_abs_err=abs(sigma2_hat - as.numeric(d$sigma2)),
                D_mse=mean((D_diag_hat - true_D_diag)^2),
                D_mae=mean(abs(D_diag_hat - true_D_diag)),
                D_full_mse=D_full_mse,
                rho_est=rho_hat, rho_abs_err=abs(rho_hat - true_rho),
                ise_macro=ise))
              } # end else (VarCorr succeeded)
            }
          }

          ## ---- bam (diagonal D, iid only) ----------------------------------
          ## bam has no corCompSymm; skip it for CS and let gamm be the R baseline.
          if ((is.null(only) || only == "bam") && D_type == "diagonal" && corr == "iid") {
            key_bam <- paste(N, pz, D_type, corr, rid, "bam", sep = "|")
            if (!(key_bam %in% done)) {
              bam_attempted <- TRUE
              cat(sprintf("randeff N=%d pz=%d D=%s C=%s repeat=%d bam ...\n",
                          N, pz, D_type, corr, rid))
              re_terms <- paste(sprintf("s(id, Zs%d, bs='re')", seq_len(pz)),
                                 collapse = " + ")
              bam_form <- as.formula(paste("y ~", smooth_terms, "+",
                                            x_terms, "+", re_terms))
              mem_before <- get_rss_mb()
              t0    <- proc.time()["elapsed"]
              fit_b <- with_time_cap(
                bam(bam_form, data = df, method = "fREML", discrete = TRUE)
              )
              t1     <- proc.time()["elapsed"]
              mem_mb <- get_rss_mb() - mem_before
              bam_timeout <- isTRUE(.fit_timed_out)

              if (is.null(fit_b)) {
                append_row(data.frame(N=N, pz=pz, D_type=D_type, corr=corr, repeat_id=rid,
                                      method="bam", converged=FALSE, timed_out=bam_timeout,
                                      time_s=NA, n_iter=NA,
                                      mem_mb=NA, beta_mse=NA, beta_mae=NA,
                                      sigma2_abs_err=NA, D_mse=NA, D_mae=NA,
                                      D_full_mse=NA, rho_est=NA, rho_abs_err=NA,
                                      ise_macro=NA))
              } else {
                n_iter_b <- if (length(fit_b$iter) > 0L) fit_b$iter else NA_integer_
                bam_post <- tryCatch({
                  newdf_b <- data.frame(t1 = t_grid)
                  newdf_b$id <- factor(rep(levels(df$id)[1], length(t_grid)),
                                        levels = levels(df$id))
                  for (j in seq_len(px)) newdf_b[[paste0("X", j)]] <- 0
                  for (j in seq_len(pz)) newdf_b[[paste0("Zs", j)]] <- 0
                  terms_b <- predict(fit_b, newdata = newdf_b, type = "terms")
                  col_idx <- grep("^s\\(t1\\)$", colnames(terms_b))
                  smooth_hat_b <- if (length(col_idx) > 0) as.numeric(terms_b[, col_idx])
                                  else rep(0, length(t_grid))
                  ise_b <- compute_ise(t_grid, smooth_hat_b, g_true)

                  # Extract D directly from smoothing parameters: D_c = sigma2 / lambda_c.
                  # Avoids gam.vcomp, which fails when the Hessian is non-PD.
                  re_sp_names  <- grep("^s\\(id,Zs", names(fit_b$sp), value = TRUE)
                  D_diag_hat_b <- if (length(re_sp_names) == pz)
                                    fit_b$sig2 / fit_b$sp[re_sp_names]
                                  else rep(NA_real_, pz)

                  sigma2_hat_b  <- fit_b$sig2
                  beta_hat_b    <- coef(fit_b)[paste0("X", seq_len(px))]
                  true_D_full_b <- diag(true_D_diag)
                  D_hat_full_b  <- diag(D_diag_hat_b)

                  list(converged=TRUE, ise=ise_b,
                       beta_mse=mean((beta_hat_b - true_beta)^2),
                       beta_mae=mean(abs(beta_hat_b - true_beta)),
                       sigma2_abs_err=abs(sigma2_hat_b - as.numeric(d$sigma2)),
                       D_mse=mean((D_diag_hat_b - true_D_diag)^2, na.rm=TRUE),
                       D_mae=mean(abs(D_diag_hat_b - true_D_diag), na.rm=TRUE),
                       D_full_mse=mean((D_hat_full_b - true_D_full_b)^2, na.rm=TRUE))
                }, error = function(e) {
                  cat(sprintf("  [bam post-fit error] %s\n", conditionMessage(e)))
                  list(converged=FALSE, ise=NA_real_,
                       beta_mse=NA_real_, beta_mae=NA_real_,
                       sigma2_abs_err=NA_real_, D_mse=NA_real_,
                       D_mae=NA_real_, D_full_mse=NA_real_)
                })

                bam_converged <- bam_post$converged
                append_row(data.frame(
                  N=N, pz=pz, D_type=D_type, corr=corr, repeat_id=rid, method="bam",
                  converged=bam_post$converged, timed_out=FALSE,
                  time_s=as.numeric(t1-t0), mem_mb=mem_mb, n_iter=n_iter_b,
                  beta_mse=bam_post$beta_mse, beta_mae=bam_post$beta_mae,
                  sigma2_abs_err=bam_post$sigma2_abs_err,
                  D_mse=bam_post$D_mse, D_mae=bam_post$D_mae,
                  D_full_mse=bam_post$D_full_mse,
                  rho_est=0.0, rho_abs_err=abs(0.0 - true_rho),
                  ise_macro=bam_post$ise))
              }
            }
          }

          ## ---- lmesp: REMOVED --------------------------------------------
          ## lmeSplines stacks one random-effect column per knot at a single
          ## group; nlme::lme is ~cubic in that dimension and never produced a
          ## usable fit. Redundant with gamm. Dropped 2026-07-06.

          ## ---- timeout bookkeeping: abort this config once a method has
          ## NEVER produced a converged fit here AND has exceeded
          ## DEFER_THRESHOLD timeouts, rather than burning the full TIME_CAP
          ## on every remaining rid of a config that is already clearly
          ## intractable for that method. A method that has converged even
          ## once is left alone regardless of how many timeouts pile up
          ## afterward -- it has proven it CAN succeed here, so later
          ## timeouts read as hard reps, not an intractable config.
          if (gamm_timeout)   gamm_timeouts   <- gamm_timeouts   + 1L
          if (bam_timeout)    bam_timeouts    <- bam_timeouts    + 1L
          if (gamm_converged) gamm_successes  <- gamm_successes  + 1L
          if (bam_converged)  bam_successes   <- bam_successes   + 1L
          gamm_stuck <- gamm_successes == 0L && gamm_timeouts > DEFER_THRESHOLD
          bam_stuck  <- bam_successes  == 0L && bam_timeouts  > DEFER_THRESHOLD
          if (gamm_stuck || bam_stuck) {
            offender <- if (gamm_stuck) "gamm" else "bam"
            n_to     <- if (gamm_stuck) gamm_timeouts else bam_timeouts
            defer_config(N, pz, D_type, corr, offender, n_to)
            break
          }

        }
      }
    }
  }
  }
  cat(sprintf("Saved %s\n", out_csv))
}

# =============================================================================
# Entry point
# =============================================================================

args     <- commandArgs(trailingOnly = TRUE)
arg_N <- NULL; arg_pz <- NULL; arg_dtype <- NULL; arg_corr <- NULL; arg_only <- NULL
i <- 1
while (i <= length(args)) {
  if      (args[i] == "--N")      { arg_N     <- as.integer(args[i + 1]); i <- i + 2 }
  else if (args[i] == "--pz")     { arg_pz    <- as.integer(args[i + 1]); i <- i + 2 }
  else if (args[i] == "--D_type") { arg_dtype <- args[i + 1];             i <- i + 2 }
  else if (args[i] == "--corr")   { arg_corr  <- args[i + 1];             i <- i + 2 }
  else if (args[i] == "--only")   { arg_only  <- args[i + 1];             i <- i + 2 }
  else { i <- i + 1 }
}
if (!is.null(arg_only) && !(arg_only %in% c("gamm", "bam")))
  stop("--only must be 'gamm' or 'bam'")

cfg         <- jsonlite::fromJSON(sprintf("%s/config_randeff.json", DATA_DIR))
N_list      <- if (!is.null(arg_N))     arg_N     else cfg$N_list
pz_list     <- if (!is.null(arg_pz))    arg_pz    else cfg$pz_list
D_type_list <- if (!is.null(arg_dtype)) arg_dtype else cfg$D_type_list
corr_list   <- if (!is.null(arg_corr))  arg_corr  else cfg$corr_structure_list
run_randeff(cfg, N_list, pz_list, D_type_list, corr_list, only = arg_only)
