# R baselines for smoothspline_comparison.
#
# Pure SS  : smooth.spline (pooled) and mgcv::gam / bam (pooled)
# Mixed SS : mgcv::gamm and bam (per-component smooths + fixed + random effects)
#
# Each spline column d is sampled on its own independent grid, and the true
# total smooth is the sum of standardized component curves. ISE is computed
# per component: the centred fitted smooth s(t_d) vs the centred standardized
# true curve f_d, integrated on a shared dense grid over t_d's observed support
# (matching the Python runner so the ISE numbers are directly comparable).
# The component curve ids and their standardization constants are read from each
# dataset JSON so the "truth" is identical to what was simulated.
#
# Run from project root:
#   Rscript simulation_experiments/smoothspline_comparison/run_r_baselines.R \
#     --type pure  --N 500
#   Rscript simulation_experiments/smoothspline_comparison/run_r_baselines.R \
#     --type mixed --N 500 --sn 3 --D_type diagonal

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

# Data directory — overridable so Python and R chains use separate dirs and
# never race on the same JSON files.
DATA_DIR <- Sys.getenv("SS_DATA_DIR", unset = "data")

TIME_CAP <- 3600

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

.fit_timed_out  <- FALSE
.last_error_msg <- NA_character_

with_time_cap <- function(expr, cap = TIME_CAP) {
  setTimeLimit(elapsed = cap, transient = TRUE)
  on.exit(setTimeLimit(elapsed = Inf), add = TRUE)
  .fit_timed_out  <<- FALSE
  .last_error_msg <<- NA_character_
  tryCatch(expr, error = function(e) {
    if (grepl("elapsed time limit", e$message, fixed = TRUE)) {
      .fit_timed_out <<- TRUE
      message(sprintf("TIMEOUT: exceeded %ds", cap))
    } else {
      .last_error_msg <<- conditionMessage(e)
      message("ERROR: ", e$message)
    }
    NULL
  })
}

# A config (N, pz, sn, D_type) that racks up too many timeouts on a method
# that has NEVER converged there is almost certainly intractable for the rest
# of its repeats too -- rather than burn the full TIME_CAP on each remaining
# rid, abort the config outright and record it so no future invocation
# re-attempts it. A method that HAS converged at least once in a config is
# left alone regardless of later timeouts: it has proven it can succeed here,
# so those read as hard reps, not an intractable config. Mirrors
# randeff_comparison/run_r_baselines.R's deferral logic.
DEFER_THRESHOLD <- 10
DEFERRED_CSV    <- "results/r_deferred.csv"

# method-specific: a deferral recorded for "bam" must not silence a separate
# --only_gamm job still making progress on the same (N, pz, sn, D_type), and
# vice versa -- that pairing is exactly what splitting the two into separate
# SLURM jobs is for.
is_config_deferred <- function(N, pz, sn, D_type, method) {
  if (!file.exists(DEFERRED_CSV)) return(FALSE)
  d <- tryCatch(read.csv(DEFERRED_CSV), error = function(e) NULL)
  if (is.null(d) || nrow(d) == 0) return(FALSE)
  any(d$N == N & d$pz == pz & d$sn == sn & d$D_type == D_type & d$method == method)
}

defer_config <- function(N, pz, sn, D_type, method, n_timeouts) {
  dir.create("results", showWarnings = FALSE)
  row <- data.frame(N = N, pz = pz, sn = sn, D_type = D_type,
                     method = method, n_timeouts = n_timeouts,
                     stringsAsFactors = FALSE)
  write.table(row, DEFERRED_CSV, sep = ",",
              col.names = !file.exists(DEFERRED_CSV),
              append = file.exists(DEFERRED_CSV),
              row.names = FALSE, quote = TRUE)
  cat(sprintf(paste("DEFERRED: N=%d pz=%d sn=%d D=%s -- %s exceeded %d",
                     "timeouts with zero converged fits, aborting remaining",
                     "reps for this configuration\n"),
              N, pz, sn, D_type, method, n_timeouts))
}

# gamm's smooth-term k is capped at K_CAP_INIT; on a non-timeout fit error
# (typically NA/NaN/Inf from an ill-conditioned, over-parameterized basis when
# sn is large relative to N) we halve k and refit, down to K_FLOOR.
K_CAP_INIT <- 500
K_FLOOR    <- 20

trapz <- function(x, y) sum(diff(x) * (head(y, -1) + tail(y, -1)) / 2)

# --- true curve library (mirrors MIXED_SS_CURVES in simscripts/LMMSS.py) -----

mixed_curves <- list(
  function(t) 2 * sin(pi * t),                                       # 0
  function(t) 2 * (3*t^2 - 2*t^3),                                   # 1  Hermite step 0→2
  function(t) 3 * exp(-15 * (t - 0.2)^2),                            # 2  Gaussian bump at 0.2
  function(t) 4 * t * (1 - t),                                       # 3
  function(t) sin(4 * pi * t),                                       # 4
  function(t) sqrt(pmax(t*(1-t), 1e-6)) * sin(2*pi*1.05/(t+0.05)), # 5
  function(t) tanh(8 * (t - 0.5)),                                   # 6
  function(t) exp(-50 * (t - 0.5)^2),                                # 7
  function(t) exp(-100*(t-0.25)^2) + 0.8*exp(-100*(t-0.75)^2),     # 8
  function(t) abs(t - 0.5)                                           # 9
)

global_curve <- function(t) 2 * sin(pi * t)
local_curve  <- function(t) 0.2*t^11*(10*(1-t))^6 + 10*(10*t)^3*(1-t)^10

# Standardized component curve, using the per-dataset constants persisted by the
# generator so the truth matches the simulated curve bit-for-bit. cid is 0-based.
norm_curve <- function(cid, mean_d, std_d) {
  raw <- mixed_curves[[cid + 1L]]
  function(t) (raw(t) - mean_d) / std_d
}

compute_ise <- function(t_grid, fitted, true) {
  fc <- fitted - mean(fitted)
  tc <- true   - mean(true)
  trapz(t_grid, (fc - tc)^2)
}

# Full-matrix D losses: Frobenius ||D̂−D||_F and Stein/entropy loss
# tr(D̂D⁻¹) − log|D̂D⁻¹| − p, capturing off-diagonal error the diagonal D_mse
# misses. Returns c(frob, stein); NA on any singularity/failure.
full_d_losses <- function(Dhat, Dtrue) {
  tryCatch({
    if (is.null(Dhat) || any(!is.finite(Dhat))) return(c(NA_real_, NA_real_))
    frob <- sqrt(sum((Dhat - Dtrue)^2))
    M    <- Dhat %*% solve(Dtrue)
    ev   <- det(M)
    stein <- if (ev > 0) sum(diag(M)) - log(ev) - nrow(Dtrue) else NA_real_
    c(frob, stein)
  }, error = function(e) c(NA_real_, NA_real_))
}

# Estimated random-effects covariance D̂ from a gamm lme fit: the `id` block's
# relative covariance times the residual variance.
gamm_Dhat <- function(fit, pz) {
  tryCatch({
    rc <- as.matrix(fit$lme$modelStruct$reStruct$id) * (fit$lme$sigma^2)
    nm <- paste0("Zs", seq_len(pz))
    rc[nm, nm, drop = FALSE]
  }, error = function(e) NULL)
}

# =============================================================================
# PURE SS
# =============================================================================

run_pure <- function(cfg, N_list, scenario_list) {
  slice_mode <- length(N_list) == 1 && length(scenario_list) == 1
  out_csv    <- if (slice_mode)
    sprintf("results/r_pure_N%d_scenario%s.csv", N_list, scenario_list)
  else
    "results/r_pure_results.csv"

  dir.create("results", showWarnings = FALSE)
  done <- list(); write_header <- TRUE
  if (file.exists(out_csv)) {
    ex  <- read.csv(out_csv)
    write_header <- FALSE
    keys <- paste(ex$N, ex$scenario, ex$repeat_id, ex$method, sep = "|")
    done <- as.list(keys)
    cat(sprintf("Resuming: %d rows done in %s\n", length(done), out_csv))
  }

  append_row <- function(row_df) {
    write.table(row_df, out_csv, sep = ",", col.names = write_header,
                append = !write_header, row.names = FALSE, quote = TRUE)
    write_header <<- FALSE
  }

  for (N in N_list) {
    for (scenario in scenario_list) {
      true_fn <- if (scenario == "global") global_curve else local_curve

      for (rid in cfg$repeat_ids) {
        path <- sprintf("%s/pure_N%d_scenario%s_repeat%d.json", DATA_DIR, N, scenario, rid)
        d    <- tryCatch(jsonlite::fromJSON(path, simplifyVector = TRUE),
                         error = function(e) { message("Missing: ", path); NULL })
        if (is.null(d)) next

        y  <- as.numeric(d$y)
        t  <- as.numeric(d$splines_vars)
        df <- data.frame(y = y, t = t)
        chk <- if (!is.null(d$data_checksum)) as.character(d$data_checksum) else ""

        # Shared dense grid over the observed support (matches the Python runner).
        t_grid <- seq(min(t), max(t), length.out = 500)
        g_true <- true_fn(t_grid)

        for (method in c("smooth.spline", "gam", "bam")) {
          key <- paste(N, scenario, rid, method, sep = "|")
          if (key %in% done) next
          cat(sprintf("pure N=%d scenario=%s repeat=%d %s ...\n", N, scenario, rid, method))

          mem_before <- get_rss_mb()
          t0  <- proc.time()["elapsed"]
          fit <- with_time_cap({
            if (method == "smooth.spline") {
              smooth.spline(t, y, all.knots = TRUE)
            } else if (method == "gam") {
              k_cr <- min(length(unique(t)), 500)
              gam(y ~ s(t, bs = "cr", k = k_cr), data = df, method = "REML")
            } else {
              k_cr <- min(length(unique(t)), 500)
              bam(y ~ s(t, bs = "cr", k = k_cr), data = df,
                  method = "fREML", discrete = TRUE)
            }
          })
          t1       <- proc.time()["elapsed"]
          mem_mb   <- get_rss_mb() - mem_before

          if (is.null(fit)) {
            append_row(data.frame(N=N, scenario=scenario, repeat_id=rid, method=method,
                                  converged=FALSE, time_s=NA, mem_mb=NA, ise_macro=NA,
                                  data_checksum=chk))
            next
          }

          fitted <- if (method == "smooth.spline")
            predict(fit, t_grid)$y
          else
            as.numeric(predict(fit, newdata = data.frame(t = t_grid)))

          ise <- compute_ise(t_grid, fitted, g_true)

          append_row(data.frame(N=N, scenario=scenario, repeat_id=rid, method=method,
                                converged=TRUE, time_s=as.numeric(t1-t0),
                                mem_mb=mem_mb, ise_macro=ise, data_checksum=chk))
        }
      }
    }
  }
  cat(sprintf("Saved %s\n", out_csv))
}

# =============================================================================
# MIXED SS
# =============================================================================

run_mixed <- function(cfg, N_list, sn_list, D_type_list, only_bam = FALSE, only_gamm = FALSE) {
  slice_mode <- length(N_list) == 1 && length(sn_list) == 1 && length(D_type_list) == 1
  # Method-suffixed like randeff_comparison's --only output: gamm and bam are
  # meant to run as separate, parallel SLURM jobs (see slurm/run_array_ss.sh),
  # and without a per-method filename they would race on the same out_csv.
  suffix  <- if (only_bam) "_bam" else if (only_gamm) "_gamm" else ""
  out_csv <- if (slice_mode)
    sprintf("results/r_mixed_N%d_sn%d_D%s%s.csv", N_list, sn_list, D_type_list, suffix)
  else
    sprintf("results/r_mixed_results%s.csv", suffix)

  dir.create("results", showWarnings = FALSE)
  done <- list(); write_header <- TRUE
  if (file.exists(out_csv)) write_header <- FALSE

  # Skip-set is seeded from EVERY existing r_mixed*.csv, not just out_csv --
  # this run may be a freshly-split --only_gamm/--only_bam job, and needs to
  # see (and skip) rows already produced by an older combined (non-split) run,
  # or by the OTHER method's separate job, rather than re-fitting them.
  existing_csvs <- Sys.glob("results/r_mixed*.csv")
  if (length(existing_csvs) > 0) {
    ex_all <- do.call(rbind, lapply(existing_csvs, function(f)
      tryCatch(read.csv(f), error = function(e) NULL)))
    if (!is.null(ex_all) && nrow(ex_all) > 0) {
      keys <- paste(ex_all$N, ex_all$sn, ex_all$D_type, ex_all$repeat_id,
                    ex_all$method, sep = "|")
      done <- as.list(unique(keys))
      cat(sprintf("Resuming: %d rows already done across %d existing file(s)\n",
                  length(done), length(existing_csvs)))
    }
  }

  append_row <- function(row_df) {
    write.table(row_df, out_csv, sep = ",", col.names = write_header,
                append = !write_header, row.names = FALSE, quote = TRUE)
    write_header <<- FALSE
  }

  pz_list <- cfg$pz_list  # fixed at 2 in the config
  px      <- cfg$px

  # Per-component ISE on a shared dense grid over each component's observed
  # support (matches the Python runner). predict_fn(newdf) returns type="terms".
  component_ise <- function(predict_fn, sn, t_df, curve_ids, curve_means,
                            curve_stds, set_id = FALSE, id_levels = NULL) {
    vapply(seq_len(sn), function(c) {
      tc   <- t_df[[c]]
      grid <- seq(min(tc), max(tc), length.out = 500)
      newdf <- as.data.frame(matrix(0, nrow = length(grid), ncol = sn))
      colnames(newdf) <- paste0("t", seq_len(sn))
      for (j in seq_len(sn)) newdf[[paste0("t", j)]] <- median(t_df[[j]])
      newdf[[paste0("t", c)]] <- grid       # s(t_c) is additive in t_c only
      for (j in seq_len(px)) newdf[[paste0("X", j)]] <- 0
      if (set_id) {
        newdf$id <- factor(rep(id_levels[1], length(grid)), levels = id_levels)
        for (j in seq_len(pz)) newdf[[paste0("Zs", j)]] <- 0
      }
      terms   <- predict_fn(newdf)
      col_idx <- grep(sprintf("^s\\(t%d\\)$", c), colnames(terms))
      if (length(col_idx) == 0) return(NA_real_)
      truth <- norm_curve(curve_ids[c], curve_means[c], curve_stds[c])(grid)
      compute_ise(grid, as.numeric(terms[, col_idx]), truth)
    }, numeric(1))
  }

  for (N in N_list) {
    for (pz in pz_list) {
      for (sn in sn_list) {
        for (D_type in D_type_list) {

          # Methods THIS job actually attempts for this D_type (bam is
          # diagonal-only). Skip the config only once every one of them is
          # deferred -- a bam deferral must not silence a still-productive
          # separate --only_gamm job on the same (N, pz, sn, D_type).
          methods_here <- character(0)
          if (!only_bam) methods_here <- c(methods_here, "gamm")
          if (!only_gamm && D_type == "diagonal") methods_here <- c(methods_here, "bam")
          config_deferred <- function() {
            length(methods_here) > 0 && all(vapply(
              methods_here, function(m) is_config_deferred(N, pz, sn, D_type, m),
              logical(1)))
          }

          if (config_deferred()) {
            cat(sprintf(paste("SKIP (deferred): N=%d pz=%d sn=%d D=%s --",
                               "previously exceeded %d timeouts\n"),
                        N, pz, sn, D_type, DEFER_THRESHOLD))
            next
          }
          gamm_timeouts <- 0L; bam_timeouts <- 0L
          gamm_successes <- 0L; bam_successes <- 0L

          for (rid in cfg$repeat_ids) {
            # Re-check each rid too, not just on config entry: a concurrently
            # running sibling job (e.g. --only_bam vs --only_gamm) may defer
            # this same config mid-run, and there is no cheaper point to
            # notice than here.
            if (config_deferred()) break
            gamm_timeout <- FALSE; gamm_converged <- FALSE
            bam_timeout  <- FALSE; bam_converged  <- FALSE
            path <- sprintf("%s/mixed_N%d_pz%d_sn%d_D%s_repeat%d.json",
                            DATA_DIR, N, pz, sn, D_type, rid)
            d <- tryCatch(jsonlite::fromJSON(path, simplifyVector = TRUE),
                          error = function(e) { message("Missing: ", path); NULL })
            if (is.null(d)) next

            y   <- as.numeric(d$y)
            ## splines_vars is N_total x sn; each column has its own grid of t values
            sv  <- matrix(as.numeric(unlist(d$splines_vars)), ncol = sn)
            X   <- matrix(as.numeric(unlist(d$X)), ncol = px)
            Z   <- matrix(as.numeric(unlist(d$Z)), ncol = pz)
            ids <- factor(as.integer(d$subject_id) + 1L)

            t_df <- as.data.frame(sv); colnames(t_df) <- paste0("t", seq_len(sn))
            X_df <- as.data.frame(X);  colnames(X_df) <- paste0("X", seq_len(px))
            Z_df <- as.data.frame(Z);  colnames(Z_df) <- paste0("Zs", seq_len(pz))
            df   <- cbind(data.frame(y = y, id = ids), t_df, X_df, Z_df)

            x_terms <- paste(paste0("X", seq_len(px)), collapse = " + ")
            k_per   <- sapply(seq_len(sn), function(c)
                              min(length(unique(t_df[[c]])), 500))
            smooth_terms <- paste(
              sprintf("s(t%d, bs='cr', k=%d)", seq_len(sn), k_per),
              collapse = " + ")

            rand_form  <- as.formula(paste("~ 0 +",
                                            paste(paste0("Zs", seq_len(pz)),
                                                  collapse = "+")))
            pd_struct  <- if (D_type == "diagonal") pdDiag(rand_form) else pdSymm(rand_form)

            true_D_full <- matrix(as.numeric(unlist(d$D)), nrow = pz)
            true_D_diag <- diag(true_D_full)
            true_beta   <- as.numeric(d$beta)

            chk         <- if (!is.null(d$data_checksum))
                             as.character(d$data_checksum) else ""
            curve_ids   <- as.integer(d$curve_ids)
            curve_means <- as.numeric(d$curve_means)
            curve_stds  <- as.numeric(d$curve_stds)

            ## ---- gamm ----------------------------------------------------
            key_gamm <- paste(N, sn, D_type, rid, "gamm", sep = "|")
            if (!only_bam && !(key_gamm %in% done)) {
              cat(sprintf("mixed N=%d pz=%d sn=%d D=%s repeat=%d gamm ...\n",
                          N, pz, sn, D_type, rid))

              gamm_form_k <- function(k_cap) {
                k_try <- sapply(seq_len(sn), function(c)
                  min(length(unique(t_df[[c]])), k_cap))
                terms_try <- paste(
                  sprintf("s(t%d, bs='cr', k=%d)", seq_len(sn), k_try),
                  collapse = " + ")
                as.formula(paste("y ~", terms_try, "+", x_terms))
              }

              k_cap     <- K_CAP_INIT
              gamm_err  <- NA_character_
              n_retries <- 0L
              mem_before <- get_rss_mb()
              t0  <- proc.time()["elapsed"]
              repeat {
                # Budget is what's LEFT of TIME_CAP, not a fresh cap per
                # attempt: with_time_cap() resets its internal timer on every
                # call, so without this a fit that only succeeds after
                # several near-cap retries could log cumulative time_s many
                # multiples of TIME_CAP while never tripping .fit_timed_out.
                remaining <- TIME_CAP - as.numeric(proc.time()["elapsed"] - t0)
                if (remaining <= 0) {
                  fit <- NULL
                  .fit_timed_out <<- TRUE
                  message(sprintf("TIMEOUT: exceeded %ds cumulative across %d retries",
                                  TIME_CAP, n_retries))
                  break
                }
                fit <- with_time_cap(
                  gamm(gamm_form_k(k_cap), random = list(id = pd_struct),
                       data = df, method = "REML",
                       control = lmeControl(opt = "optim",
                                            maxIter = 200, msMaxIter = 200)),
                  cap = remaining
                )
                if (is.null(fit) && !.fit_timed_out) gamm_err <- .last_error_msg
                if (!is.null(fit) || .fit_timed_out || k_cap <= K_FLOOR) break
                k_cap     <- max(floor(k_cap / 2), K_FLOOR)
                n_retries <- n_retries + 1L
                cat(sprintf("  gamm error, retrying with k=%d (attempt %d): %s\n",
                            k_cap, n_retries, gamm_err))
              }
              t1     <- proc.time()["elapsed"]
              mem_mb <- get_rss_mb() - mem_before
              gamm_timeout <- isTRUE(.fit_timed_out)

              if (is.null(fit)) {
                append_row(data.frame(N=N, pz=pz, sn=sn, D_type=D_type,
                                      repeat_id=rid, method="gamm",
                                      converged=FALSE, time_s=NA, mem_mb=NA,
                                      n_iter=NA,
                                      beta_mse=NA, beta_mae=NA,
                                      sigma2_abs_err=NA, D_mse=NA, D_mae=NA,
                                      D_frob=NA, D_stein=NA,
                                      ise_macro=NA, ise_median=NA, data_checksum=chk,
                                      error_msg=gamm_err, k_final=k_cap,
                                      k_retries=n_retries))
              } else {
                ise_comps <- component_ise(
                  function(nd) predict(fit$gam, newdata = nd, type = "terms"),
                  sn, t_df, curve_ids, curve_means, curve_stds)
                ise     <- mean(ise_comps, na.rm = TRUE)
                ise_med <- median(ise_comps, na.rm = TRUE)

                converged  <- is.matrix(fit$lme$apVar)
                gamm_converged <- converged
                vc         <- VarCorr(fit$lme)
                D_diag_hat <- as.numeric(vc[paste0("Zs", seq_len(pz)), "StdDev"])^2
                sigma2_hat <- as.numeric(vc["Residual", "StdDev"])^2
                beta_hat   <- coef(fit$gam)[paste0("X", seq_len(px))]
                b_err <- list(mse = mean((beta_hat - true_beta)^2),
                              mae = mean(abs(beta_hat - true_beta)))
                D_err <- list(mse = mean((D_diag_hat - true_D_diag)^2),
                              mae = mean(abs(D_diag_hat - true_D_diag)))
                d_full <- full_d_losses(gamm_Dhat(fit, pz), true_D_full)

                append_row(data.frame(
                  N=N, pz=pz, sn=sn, D_type=D_type, repeat_id=rid, method="gamm",
                  converged=converged, time_s=as.numeric(t1-t0), mem_mb=mem_mb,
                  n_iter=if (!is.null(fit$lme$numIter)) as.integer(fit$lme$numIter) else NA_integer_,
                  beta_mse=b_err$mse, beta_mae=b_err$mae,
                  sigma2_abs_err=abs(sigma2_hat - as.numeric(d$sigma2)),
                  D_mse=D_err$mse, D_mae=D_err$mae,
                  D_frob=d_full[1], D_stein=d_full[2],
                  ise_macro=ise, ise_median=ise_med, data_checksum=chk,
                  error_msg=gamm_err, k_final=k_cap, k_retries=n_retries
                ))
              }
            }

            ## ---- bam (diagonal D only) -----------------------------------
            ## bam's bs="re" infrastructure represents random effects as
            ## independent smooth terms — equivalent to a diagonal D — so we
            ## only run it when the simulated D is diagonal.
            if (D_type == "diagonal" && !only_gamm) {
              key_bam <- paste(N, sn, D_type, rid, "bam", sep = "|")
              if (!(key_bam %in% done)) {
                cat(sprintf("mixed N=%d pz=%d sn=%d D=%s repeat=%d bam ...\n",
                            N, pz, sn, D_type, rid))

                re_terms <- paste(sprintf("s(id, Zs%d, bs='re')", seq_len(pz)),
                                   collapse = " + ")
                bam_form <- as.formula(paste("y ~", smooth_terms, "+",
                                              x_terms, "+", re_terms))

                mem_before <- get_rss_mb()
                t0  <- proc.time()["elapsed"]
                # discrete = FALSE for a fair speed comparison: bam otherwise
                # uses covariate discretization (a computational speedup on the
                # same basis) that our method does not, so timings weren't
                # apples-to-apples. Same basis dim (k = #unique t), dense compute.
                fit_b <- with_time_cap(
                  bam(bam_form, data = df, method = "fREML", discrete = FALSE)
                )
                t1     <- proc.time()["elapsed"]
                mem_mb <- get_rss_mb() - mem_before
                bam_timeout <- isTRUE(.fit_timed_out)

                if (is.null(fit_b)) {
                  append_row(data.frame(N=N, pz=pz, sn=sn, D_type=D_type,
                                        repeat_id=rid, method="bam",
                                        converged=FALSE, time_s=NA, mem_mb=NA,
                                        n_iter=NA,
                                        beta_mse=NA, beta_mae=NA,
                                        sigma2_abs_err=NA, D_mse=NA, D_mae=NA,
                                        D_frob=NA, D_stein=NA,
                                        ise_macro=NA, ise_median=NA, data_checksum=chk,
                                        error_msg=NA_character_,
                                        k_final=NA_integer_,
                                        k_retries=NA_integer_))
                } else {
                  bam_converged <- TRUE
                  ise_comps_b <- component_ise(
                    function(nd) predict(fit_b, newdata = nd, type = "terms"),
                    sn, t_df, curve_ids, curve_means, curve_stds,
                    set_id = TRUE, id_levels = levels(df$id))
                  ise_b     <- mean(ise_comps_b, na.rm = TRUE)
                  ise_med_b <- median(ise_comps_b, na.rm = TRUE)

                  # Extract D directly from smoothing parameters: D_c = sigma2 / lambda_c.
                  # This avoids gam.vcomp, which fails when the Hessian is non-PD.
                  re_sp_names  <- grep("^s\\(id,Zs", names(fit_b$sp), value = TRUE)
                  D_diag_hat_b <- if (length(re_sp_names) == pz)
                                    fit_b$sig2 / fit_b$sp[re_sp_names]
                                  else rep(NA_real_, pz)

                  sigma2_hat_b <- fit_b$sig2
                  beta_hat_b   <- coef(fit_b)[paste0("X", seq_len(px))]
                  b_err_b <- list(mse = mean((beta_hat_b - true_beta)^2),
                                  mae = mean(abs(beta_hat_b - true_beta)))
                  D_err_b <- list(
                    mse = mean((D_diag_hat_b - true_D_diag)^2, na.rm = TRUE),
                    mae = mean(abs(D_diag_hat_b - true_D_diag), na.rm = TRUE)
                  )
                  # bam's re-smooth represents a diagonal D̂.
                  d_full_b <- full_d_losses(diag(D_diag_hat_b, nrow = pz),
                                            true_D_full)

                  append_row(data.frame(
                    N=N, pz=pz, sn=sn, D_type=D_type, repeat_id=rid, method="bam",
                    converged=TRUE, time_s=as.numeric(t1-t0), mem_mb=mem_mb,
                    n_iter=if (length(fit_b$iter) > 0L) fit_b$iter else NA_integer_,
                    beta_mse=b_err_b$mse, beta_mae=b_err_b$mae,
                    sigma2_abs_err=abs(sigma2_hat_b - as.numeric(d$sigma2)),
                    D_mse=D_err_b$mse, D_mae=D_err_b$mae,
                    D_frob=d_full_b[1], D_stein=d_full_b[2],
                    ise_macro=ise_b, ise_median=ise_med_b, data_checksum=chk,
                    error_msg=NA_character_, k_final=NA_integer_,
                    k_retries=NA_integer_
                  ))
                }
              }
            }

            ## ---- lmesp: REMOVED ---------------------------------------------
            ## lmeSplines represents each smooth as one random-effect column per
            ## knot (= unique t), all sn blocks stacked at a single top-level
            ## group, so nlme::lme factors a dense sn*K_c structure (~cubic cost).
            ## It only fits while sn*K_c is a few hundred; every cell here (sn>=5,
            ## K_c=81-175 -> 400-1750 cols) times out, producing all-NA rows. It
            ## is also redundant with gamm (penalized spline via lme). Dropped.

            ## ---- timeout bookkeeping: abort this config once a method has
            ## NEVER produced a converged fit here AND has exceeded
            ## DEFER_THRESHOLD timeouts. A method that has converged even once
            ## is left alone regardless of how many timeouts pile up
            ## afterward -- it has proven it CAN succeed here.
            if (gamm_timeout)   gamm_timeouts   <- gamm_timeouts   + 1L
            if (bam_timeout)    bam_timeouts    <- bam_timeouts    + 1L
            if (gamm_converged) gamm_successes  <- gamm_successes  + 1L
            if (bam_converged)  bam_successes   <- bam_successes   + 1L
            gamm_stuck <- gamm_successes == 0L && gamm_timeouts > DEFER_THRESHOLD
            bam_stuck  <- bam_successes  == 0L && bam_timeouts  > DEFER_THRESHOLD
            if (gamm_stuck || bam_stuck) {
              offender <- if (gamm_stuck) "gamm" else "bam"
              n_to     <- if (gamm_stuck) gamm_timeouts else bam_timeouts
              defer_config(N, pz, sn, D_type, offender, n_to)
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

args      <- commandArgs(trailingOnly = TRUE)
arg_type  <- NULL; arg_N <- NULL; arg_sn <- NULL; arg_dtype <- NULL
arg_reps  <- NULL
arg_only_bam <- FALSE
arg_only_gamm <- FALSE
i <- 1
while (i <= length(args)) {
  if      (args[i] == "--type")      { arg_type  <- args[i + 1];             i <- i + 2 }
  else if (args[i] == "--N")         { arg_N     <- as.integer(args[i + 1]); i <- i + 2 }
  else if (args[i] == "--sn")        { arg_sn    <- as.integer(args[i + 1]); i <- i + 2 }
  else if (args[i] == "--D_type")    { arg_dtype <- args[i + 1];             i <- i + 2 }
  else if (args[i] == "--reps")      { arg_reps  <- as.integer(args[i + 1]); i <- i + 2 }
  else if (args[i] == "--only_bam")  { arg_only_bam <- TRUE;                 i <- i + 1 }
  else if (args[i] == "--only_gamm") { arg_only_gamm <- TRUE;                i <- i + 1 }
  else { i <- i + 1 }
}

if (is.null(arg_type)) stop("--type pure|mixed is required")

if (arg_type == "pure") {
  cfg         <- jsonlite::fromJSON(sprintf("%s/config_pure.json", DATA_DIR))
  N_list      <- if (!is.null(arg_N)) arg_N else cfg$N_list
  scenario_list <- cfg$scenario_list
  if (!is.null(arg_reps)) cfg$repeat_ids <- head(cfg$repeat_ids, arg_reps)
  run_pure(cfg, N_list, scenario_list)

} else {
  cfg         <- jsonlite::fromJSON(sprintf("%s/config_mixed.json", DATA_DIR))
  N_list      <- if (!is.null(arg_N))     arg_N     else cfg$N_list
  sn_list     <- if (!is.null(arg_sn))    arg_sn    else cfg$sn_list
  D_type_list <- if (!is.null(arg_dtype)) arg_dtype else cfg$D_type_list
  if (!is.null(arg_reps)) cfg$repeat_ids <- head(cfg$repeat_ids, arg_reps)
  run_mixed(cfg, N_list, sn_list, D_type_list, only_bam = arg_only_bam,
            only_gamm = arg_only_gamm)
}
