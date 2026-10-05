#' @title TumorFS: R Interface to Multi-Algorithm Feature Selection
#'
#' @description
#' R wrapper for the TumorFS Python package via reticulate.
#' Provides feature selection, ensemble ranking, and downstream
#' classification for RNA-seq tumour data.
#'
#' @importFrom reticulate import py_install use_condaenv
#' @name tumorfs-package
NULL

# ── Internal Python module handle ────────────────────────────────────────────
.tfs <- new.env(parent = emptyenv())

#' Initialise the TumorFS Python backend
#'
#' Must be called once per session before using any other tumorfs function.
#' Optionally specify a conda / virtualenv name that has tumorfs installed.
#'
#' @param envname  Name of the conda / virtual environment (default: "tumorfs").
#' @param method   One of \code{"conda"} or \code{"virtualenv"}.
#' @param install  If \code{TRUE}, attempt to \code{pip install} tumorfs
#'                 into the environment automatically.
#'
#' @export
#' @examples
#' \dontrun{
#' tfs_init()
#' }
tfs_init <- function(envname = "tumorfs", method = "conda", install = FALSE) {
  if (method == "conda") {
    reticulate::use_condaenv(envname, required = TRUE)
  } else {
    reticulate::use_virtualenv(envname, required = TRUE)
  }

  if (install) {
    reticulate::py_install(
      "git+https://github.com/<your-username>/TumorFS.git",
      envname = envname, method = method, pip = TRUE
    )
  }

  .tfs$tfs   <- reticulate::import("tumorfs")
  .tfs$ready <- TRUE
  message("TumorFS Python backend ready.")
  invisible(.tfs$tfs)
}

.check_init <- function() {
  if (!isTRUE(.tfs$ready)) {
    stop("Call tfs_init() first to initialise the Python backend.")
  }
}


# ── Data loading ─────────────────────────────────────────────────────────────

#' Load RNA-seq expression data
#'
#' @param file_path   Path to expression CSV (genes × samples — wide format,
#'                    or samples × genes — tidy format).
#' @param layout      \code{"wide"} (default) or \code{"tidy"}.
#' @param test_size   Fraction for test split (default 0.2).
#' @param balance     Downsample majority class (default TRUE).
#' @param random_state Integer seed.
#'
#' @return A named list with elements \code{X_train}, \code{X_test},
#'   \code{y_train}, \code{y_test}, \code{gene_names}.
#'
#' @export
#' @examples
#' \dontrun{
#' tfs_init()
#' dat <- tfs_load("expression.csv")
#' }
tfs_load <- function(file_path,
                     layout       = "wide",
                     test_size    = 0.2,
                     balance      = TRUE,
                     random_state = 42L) {
  .check_init()
  loader <- .tfs$tfs$DataLoader(
    file_path    = file_path,
    layout       = layout,
    test_size    = test_size,
    balance      = balance,
    random_state = random_state
  )
  result <- loader$load()
  list(
    X_train    = reticulate::py_to_r(result[[1]]),
    X_test     = reticulate::py_to_r(result[[2]]),
    y_train    = reticulate::py_to_r(result[[3]]),
    y_test     = reticulate::py_to_r(result[[4]]),
    gene_names = reticulate::py_to_r(result[[5]])
  )
}


# ── Feature selection ─────────────────────────────────────────────────────────

#' Run a single feature selector
#'
#' @param X_train  Samples × genes data frame (training split).
#' @param y_train  Integer label vector.
#' @param method   One of the 22 supported selectors
#'                 (e.g. \code{"lightgbm"}, \code{"relieff"}, \code{"lasso"}).
#' @param n_features  Number of features to select.
#' @param random_state Seed.
#' @param ...      Extra arguments forwarded to the Python selector.
#'
#' @return Character vector of selected gene names.
#'
#' @export
tfs_select <- function(X_train, y_train,
                       method       = "lightgbm",
                       n_features   = 100L,
                       random_state = 42L,
                       ...) {
  .check_init()
  X_py <- reticulate::r_to_py(X_train)
  y_py <- reticulate::r_to_py(y_train)
  sel <- .tfs$tfs$FeatureSelector(
    method       = method,
    n_features   = as.integer(n_features),
    random_state = as.integer(random_state),
    ...
  )
  sel$fit(X_py, y_py)
  reticulate::py_to_r(sel$selected_features_)
}


# ── Ensemble ranking ──────────────────────────────────────────────────────────

#' Run ensemble feature ranking across multiple selectors
#'
#' @param X_train   Samples × genes data frame.
#' @param y_train   Label vector.
#' @param methods   Character vector of selector names.
#' @param n_features  Number of top features each selector considers.
#' @param strategy  \code{"rank_aggregation"} (default), \code{"vote"}, or
#'                  \code{"weighted"}.
#' @param random_state Seed.
#'
#' @return A named list:
#'   \item{top_features}{Ordered character vector of gene names.}
#'   \item{ranking_table}{Data frame with ensemble scores and per-method ranks.}
#'
#' @export
#' @examples
#' \dontrun{
#' tfs_init()
#' dat <- tfs_load("expression.csv")
#' res <- tfs_ensemble(dat$X_train, dat$y_train,
#'                     methods = c("lightgbm", "relieff", "lasso"),
#'                     n_features = 100L)
#' head(res$top_features, 20)
#' }
tfs_ensemble <- function(X_train, y_train,
                         methods      = c("lightgbm", "xgboost", "relieff", "lasso"),
                         n_features   = 200L,
                         strategy     = "rank_aggregation",
                         random_state = 42L) {
  .check_init()
  X_py <- reticulate::r_to_py(X_train)
  y_py <- reticulate::r_to_py(y_train)

  ranker <- .tfs$tfs$EnsembleRanker(
    methods      = as.list(methods),
    n_features   = as.integer(n_features),
    strategy     = strategy,
    random_state = as.integer(random_state)
  )
  ranker$fit(X_py, y_py)

  list(
    top_features  = reticulate::py_to_r(ranker$top_features_),
    ranking_table = reticulate::py_to_r(ranker$get_ranking_table())
  )
}


# ── Classification ────────────────────────────────────────────────────────────

#' Evaluate a classifier on selected genes via cross-validation
#'
#' @param X_train    Samples × genes data frame (full feature set).
#' @param y_train    Label vector.
#' @param genes      Character vector of gene names to use.
#' @param model      Classifier name (e.g. \code{"svm"}, \code{"lightgbm"}).
#' @param cv         Number of CV folds.
#' @param scoring    Character vector of sklearn scoring strings.
#' @param random_state Seed.
#'
#' @return Data frame of per-fold metrics.
#'
#' @export
tfs_classify <- function(X_train, y_train,
                         genes,
                         model        = "svm",
                         cv           = 5L,
                         scoring      = c("accuracy", "f1_macro"),
                         random_state = 42L) {
  .check_init()
  X_sub <- X_train[, genes, drop = FALSE]
  X_py  <- reticulate::r_to_py(X_sub)
  y_py  <- reticulate::r_to_py(y_train)

  clf <- .tfs$tfs$TumorClassifier(
    model        = model,
    cv           = as.integer(cv),
    scoring      = as.list(scoring),
    random_state = as.integer(random_state)
  )
  results <- clf$evaluate(X_py, y_py)
  reticulate::py_to_r(results)
}


# ── Full pipeline ─────────────────────────────────────────────────────────────

#' Run the complete TumorFS pipeline from a YAML config
#'
#' @param config_path  Path to a YAML configuration file.
#'
#' @return Invisibly, the pipeline Python object (access \code{$summary_}).
#'
#' @export
tfs_run_pipeline <- function(config_path) {
  .check_init()
  pipe <- .tfs$tfs$TumorFSPipeline(config_path)
  pipe$run()
  message("Pipeline complete.  Results saved to the configured output directory.")
  invisible(pipe)
}
