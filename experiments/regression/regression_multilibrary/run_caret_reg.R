#!/usr/bin/env Rscript
# Train one caret regression model on shared CSV matrices; write predictions JSON.
# Args: --train_x --train_y --test_x --test_y --learner --out --seed --n_train

suppressPackageStartupMessages({
  library(caret)
  library(rpart)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name, default = NULL) {
  i <- match(name, args)
  if (is.na(i)) return(default)
  args[[i + 1]]
}

train_x_path <- get_arg("--train_x")
train_y_path <- get_arg("--train_y")
test_x_path <- get_arg("--test_x")
test_y_path <- get_arg("--test_y")
learner <- get_arg("--learner")
out_path <- get_arg("--out")
seed <- as.integer(get_arg("--seed", "42"))
n_train <- as.integer(get_arg("--n_train", "NA"))

set.seed(seed)
Sys.setenv(OMP_NUM_THREADS = "1", MKL_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1")

Xtr <- as.matrix(read.csv(train_x_path, header = FALSE))
Xte <- as.matrix(read.csv(test_x_path, header = FALSE))
ytr <- as.numeric(read.csv(train_y_path, header = FALSE)[[1]])
yte <- as.numeric(read.csv(test_y_path, header = FALSE)[[1]])
if (is.na(n_train)) n_train <- nrow(Xtr)
colnames(Xtr) <- paste0("f", seq_len(ncol(Xtr)))
colnames(Xte) <- colnames(Xtr)

train_df <- data.frame(Xtr, .y = ytr, check.names = FALSE)
tc <- trainControl(method = "none", allowParallel = FALSE)

fit <- NULL
method <- NULL
tune <- NULL
backend_args <- list()
status <- "ok"
error <- NULL
t_train0 <- proc.time()[[3]]

tryCatch({
  if (learner == "linear_regression") {
    method <- "lm"
    # caret lm tunes intercept TRUE/FALSE; keep intercept to match other stacks
    tune <- data.frame(intercept = TRUE)
    fit <- train(.y ~ ., data = train_df, method = method, trControl = tc,
                 tuneGrid = tune, metric = "RMSE")
  } else if (learner == "ridge_regression") {
    # caret glmnet with alpha=0 (ridge). Data already standardized in Python.
    # Align lambda to sklearn Ridge(alpha=1) under glmnet's (1/(2n)) SSE scaling:
    # alpha_sklearn ~= 2 * n * lambda_glmnet  =>  lambda = 1 / (2 n).
    method <- "glmnet"
    lam <- 1.0 / (2.0 * as.numeric(n_train))
    tune <- data.frame(alpha = 0, lambda = lam)
    backend_args <- list(standardize = FALSE, lambda_formula = "1/(2*n_train)")
    fit <- train(.y ~ ., data = train_df, method = method, trControl = tc,
                 tuneGrid = tune, metric = "RMSE",
                 preProcess = NULL, standardize = FALSE)
  } else if (learner == "regression_tree") {
    method <- "rpart2"
    tune <- data.frame(maxdepth = 4)
    backend_args <- list(minsplit = 5, minbucket = 1, cp = 1e-8, xval = 0)
    fit <- train(.y ~ ., data = train_df, method = method, trControl = tc,
                 tuneGrid = tune, metric = "RMSE",
                 control = rpart.control(minsplit = 5, minbucket = 1, cp = 1e-8, xval = 0))
  } else if (learner == "gradient_boosting") {
    method <- "gbm"
    tune <- data.frame(n.trees = 40, interaction.depth = 2, shrinkage = 0.1, n.minobsinnode = 1)
    backend_args <- list(bag.fraction = 1.0, train.fraction = 1.0, distribution = "gaussian")
    fit <- train(.y ~ ., data = train_df, method = method, trControl = tc,
                 tuneGrid = tune, metric = "RMSE", verbose = FALSE,
                 bag.fraction = 1.0, train.fraction = 1.0, distribution = "gaussian")
  } else {
    stop(paste("unknown learner", learner))
  }
}, error = function(e) {
  status <<- "failed"
  error <<- conditionMessage(e)
})

train_ms <- (proc.time()[[3]] - t_train0) * 1000

preds <- NULL
pred_ms <- NA_real_
r2 <- NA_real_
if (identical(status, "ok") && !is.null(fit)) {
  t_pred0 <- proc.time()[[3]]
  preds <- as.numeric(predict(fit, newdata = data.frame(Xte, check.names = FALSE)))
  pred_ms <- (proc.time()[[3]] - t_pred0) * 1000
  ss_res <- sum((yte - preds)^2)
  ss_tot <- sum((yte - mean(yte))^2)
  r2 <- if (ss_tot > 0) 1 - ss_res / ss_tot else NA_real_
}

payload <- list(
  status = status,
  error = error,
  learner = learner,
  method = method,
  tune = tune,
  backend_args = backend_args,
  seed = seed,
  n_train = n_train,
  train_ms = unname(train_ms),
  predict_ms = unname(pred_ms),
  r2 = unname(r2),
  y_pred = preds
)
write_json(payload, out_path, auto_unbox = TRUE, digits = 12, null = "null")
cat(sprintf("caret_reg %s status=%s r2=%s\n", learner, status, ifelse(is.na(r2), "NA", sprintf("%.4f", r2))))
