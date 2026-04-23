# Compute true posterior co-clustering matrix across all burned-in samples
# from each H-DP pEx seed, then average across seeds and export.
args <- commandArgs(trailingOnly = TRUE)
setwd(if (length(args) >= 1) args[[1]] else ".")

in_dir <- "results/sbm_explainer/hdp_pex"
seeds <- 0:4
burn_in <- 2000
n_iter  <- 4000

# Helper — no need to source full pex-sbm.R for this
vec2mat <- function(z) {
  V <- length(z); H <- max(z)
  M <- matrix(0, V, H)
  for (v in 1:V) M[v, z[v]] <- 1
  M
}
pr_cc <- function(z_post) {
  V <- nrow(z_post); T <- ncol(z_post)
  c <- matrix(0, V, V)
  for (t in 1:T) { Z <- vec2mat(z_post[, t]); c <- c + Z %*% t(Z) }
  c / T
}

C_seeds <- list()
for (s in seeds) {
  d <- readRDS(file.path(in_dir, sprintf("seed%d.rds", s)))
  z_burned <- d$z_post[, (burn_in + 1):n_iter, drop = FALSE]
  C <- pr_cc(z_burned)
  C_seeds[[length(C_seeds) + 1]] <- C
  cat(sprintf("seed %d: C diag mean = %.3f, off-diag mean = %.3f\n",
              s, mean(diag(C)),
              mean(C[upper.tri(C)])))
}
C_avg <- Reduce("+", C_seeds) / length(C_seeds)
write.table(C_avg, file.path(in_dir, "C_post.csv"),
            sep = ",", row.names = FALSE, col.names = FALSE)
cat("Saved C_post.csv\n")
