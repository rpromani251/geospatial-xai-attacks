# H-DP pEx-SBM (Durante et al. 2025) on the Gauss+LIME fingerprint graph.
#
# Calls pex_sbm() from papers/pexsbm/pex-sbm.R. Inputs are produced by
# scripts/sbm_explainer/export_hdp_inputs.py (adjacency reordered so
# layer-1 nodes come first, then layer-2).
#
# Runs N_SEEDS Gibbs chains, saves posterior z samples, posterior
# co-clustering P_post, and the VI-optimal point estimate z_hat per seed.

suppressMessages({
  library(mcclust)
  library(mcclust.ext)
})

args <- commandArgs(trailingOnly = TRUE)
REPO   <- if (length(args) >= 1) args[[1]] else "."
setwd(REPO)

source("papers/pexsbm/pex-sbm.R")

in_dir  <- "results/sbm_explainer/hdp_pex"
Y <- as.matrix(read.csv(file.path(in_dir, "adjacency.csv"), header = FALSE))
Y <- matrix(as.numeric(Y), nrow = nrow(Y))
N_vec <- as.integer(read.table(file.path(in_dir, "N.csv"))[, 1])
cat(sprintf("Loaded Y = %d x %d, N = [%s]\n",
            nrow(Y), ncol(Y), paste(N_vec, collapse = ", ")))

# Hyperparameters (Durante et al. Section 4 defaults; matches
# simulations_HDP.R line 40)
n_iter  <- 4000
burn_in <- 2000
a <- 1; b <- 1
theta  <- 0.5
theta0 <- 4
N_SEEDS <- 5
SEEDS <- 0:(N_SEEDS - 1)

z_hats <- matrix(0, nrow = nrow(Y), ncol = N_SEEDS)
n_clusters_map <- numeric(N_SEEDS)
n_clusters_mean <- numeric(N_SEEDS)

for (si in seq_along(SEEDS)) {
  s <- SEEDS[si]
  set.seed(s)
  cat(sprintf("\n=== seed %d ===\n", s))
  t0 <- Sys.time()
  model <- pex_sbm(Y, n_iter, N_vec, a, b, theta, theta0, prior = "HDP")
  dt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  cat(sprintf("  sampler: %.1fs\n", dt))

  z_post <- model[[1]]
  z_burned <- z_post[, (burn_in + 1):n_iter, drop = FALSE]
  P_post <- pr_cc(z_burned)
  mv <- minVI(P_post, method = "avg", max.k = 20)
  z_hat <- mv$cl
  z_hats[, si] <- z_hat
  n_clusters_map[si] <- length(unique(z_hat))
  n_clusters_mean[si] <- mean(apply(z_burned, 2, function(v) length(unique(v))))
  cat(sprintf("  z_hat unique clusters: %d\n", n_clusters_map[si]))
  cat(sprintf("  posterior mean # clusters: %.2f\n", n_clusters_mean[si]))

  saveRDS(list(
    z_hat = z_hat, z_post = z_post, P_post = P_post,
    n_iter = n_iter, burn_in = burn_in,
    theta = theta, theta0 = theta0, seed = s, wall_seconds = dt
  ), file.path(in_dir, sprintf("seed%d.rds", s)))
}

write.table(z_hats, file.path(in_dir, "z_hats.csv"),
            sep = ",", row.names = FALSE, col.names = FALSE)
write.table(
  data.frame(seed = SEEDS,
             n_clusters_map = n_clusters_map,
             n_clusters_post_mean = n_clusters_mean),
  file.path(in_dir, "cluster_counts.csv"),
  sep = ",", row.names = FALSE
)
cat("\nSaved z_hats.csv, cluster_counts.csv, seed*.rds\n")
