# Sweep H-DP pEx over theta0 (base-measure concentration) to see whether
# lower theta0 produces fewer inferred clusters and higher layer NMI on
# the Gauss+LIME fingerprint graph.
#
# Low theta0 → DP on the global profile measure favours fewer new profiles
# → expect K_inferred to drop, NMI vs binary layer may rise.

suppressMessages({
  library(mcclust); library(mcclust.ext); library(forcats)
})

args <- commandArgs(trailingOnly = TRUE)
setwd(if (length(args) >= 1) args[[1]] else ".")
source("papers/pexsbm/pex-sbm.R")

in_dir <- "results/sbm_explainer/hdp_pex"
Y <- matrix(as.numeric(as.matrix(read.csv(file.path(in_dir, "adjacency.csv"), header = FALSE))),
            nrow = 200)
N_vec <- as.integer(read.table(file.path(in_dir, "N.csv"))[, 1])

THETA0_GRID <- c(0.25, 0.5, 1.0, 2.0)
THETA <- 0.5
SEEDS <- 0:2
n_iter  <- 2500
burn_in <- 1500

rows <- list()
z_hat_store <- list()
for (t0 in THETA0_GRID) {
  for (s in SEEDS) {
    tag <- sprintf("th0=%.2f_seed=%d", t0, s)
    cat(sprintf("\n── %s ──\n", tag))
    set.seed(s)
    tstart <- Sys.time()
    model <- pex_sbm(Y, n_iter, N_vec, 1, 1, THETA, t0, prior = "HDP")
    dt <- as.numeric(difftime(Sys.time(), tstart, units = "secs"))
    z_post <- model[[1]]
    z_b <- z_post[, (burn_in + 1):n_iter, drop = FALSE]
    P <- pr_cc(z_b); z_hat <- minVI(P, method = "avg", max.k = 20)$cl
    K_hat <- length(unique(z_hat))
    K_post_mean <- mean(apply(z_b, 2, function(v) length(unique(v))))
    cat(sprintf("  time=%.0fs  K_hat=%d  K_post_mean=%.2f\n", dt, K_hat, K_post_mean))
    rows[[length(rows) + 1]] <- data.frame(
      theta0 = t0, theta = THETA, seed = s,
      K_hat = K_hat, K_post_mean = K_post_mean,
      wall_s = dt
    )
    z_hat_store[[tag]] <- z_hat
  }
}
df <- do.call(rbind, rows)
write.table(df, file.path(in_dir, "sweep_rows.csv"), sep = ",",
            row.names = FALSE)

# Save all z_hats as one matrix (rows = nodes, cols = cells in sweep order)
cols <- names(z_hat_store)
zh_mat <- do.call(cbind, z_hat_store)
colnames(zh_mat) <- cols
write.table(zh_mat, file.path(in_dir, "sweep_zhats.csv"),
            sep = ",", row.names = FALSE)
cat("\nSaved sweep_rows.csv, sweep_zhats.csv\n")
