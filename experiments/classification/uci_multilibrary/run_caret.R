#!/usr/bin/env Rscript
# Train one caret model on shared CSV matrices; write predictions JSON.
# Args: --train_x --train_y --test_x --test_y --learner --out --seed --n_features

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
# --seed is the model_seed (not the data split_seed).
seed <- as.integer(get_arg("--seed", "42"))
n_features <- as.integer(get_arg("--n_features"))

set.seed(seed)
# Attempt single-thread
Sys.setenv(OMP_NUM_THREADS = "1", MKL_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1")

Xtr <- as.matrix(read.csv(train_x_path, header = FALSE))
Xte <- as.matrix(read.csv(test_x_path, header = FALSE))
ytr <- as.integer(read.csv(train_y_path, header = FALSE)[[1]])
yte <- as.integer(read.csv(test_y_path, header = FALSE)[[1]])
if (is.na(n_features)) n_features <- ncol(Xtr)
colnames(Xtr) <- paste0("f", seq_len(ncol(Xtr)))
colnames(Xte) <- colnames(Xtr)

# caret prefers non-numeric class names; map 0->Class0, 1->Class1, ...
to_class <- function(y) paste0("Class", as.character(y))
from_class <- function(f) as.integer(sub("^Class", "", as.character(f)))
levels_all <- paste0("Class", sort(unique(c(ytr, yte))))
ytr_f <- factor(to_class(ytr), levels = levels_all)
yte_f <- factor(to_class(yte), levels = levels_all)
train_df <- data.frame(Xtr, .y = ytr_f, check.names = FALSE)

tc <- trainControl(method = "none", classProbs = TRUE, allowParallel = FALSE)

t_prep <- proc.time()[[3]]
# data already prepared by Python; prep here is just frame assembly
prep_ms <- (proc.time()[[3]] - t_prep) * 1000

fit <- NULL
method <- NULL
tune <- NULL
backend_args <- list()
status <- "ok"
error <- NULL
t_train0 <- proc.time()[[3]]

tryCatch({
  if (learner == "decision_tree") {
    method <- "rpart"
    # Grow deep tree then prune weakly: cp near 0, maxdepth=8 to align with pgml/sklearn depth
    tune <- data.frame(cp = 1e-8)
    backend_args <- list(maxdepth = 8, minsplit = 2, minbucket = 1, xval = 0)
    fit <- train(.y ~ ., data = train_df, method = method, trControl = tc, tuneGrid = tune,
                 metric = "Accuracy",
                 control = rpart.control(maxdepth = 8, minsplit = 2, minbucket = 1, cp = 1e-8, xval = 0))
  } else if (learner == "bagged_trees") {
    method <- "treebag"
    # treebag has no tune parameters in caret
    fit <- train(.y ~ ., data = train_df, method = method, trControl = tc,
                 metric = "Accuracy", nbagg = 15,
                 control = rpart.control(maxdepth = 8, minsplit = 2, minbucket = 1, cp = 1e-8, xval = 0))
  } else if (learner == "gradient_boosting") {
    method <- "gbm"
    # n.trees, interaction.depth, shrinkage, n.minobsinnode
    # bag.fraction=1, train.fraction=1 — no internal holdout / early stopping on test
    tune <- data.frame(n.trees = 30, interaction.depth = 3, shrinkage = 0.1, n.minobsinnode = 1)
    fit <- train(.y ~ ., data = train_df, method = method, trControl = tc, tuneGrid = tune,
                 metric = "Accuracy", verbose = FALSE,
                 bag.fraction = 1.0, train.fraction = 1.0)
  } else if (learner == "adaboost") {
    method <- "AdaBoost.M1"
    # mfinal, maxdepth, coeflearn
    tune <- data.frame(mfinal = 30, maxdepth = 1, coeflearn = "Breiman")
    fit <- train(.y ~ ., data = train_df, method = method, trControl = tc, tuneGrid = tune,
                 metric = "Accuracy")
  } else if (learner == "linear_svm") {
    method <- "svmLinear"
    # kernlab vanilladot; C cost
    tune <- data.frame(C = 1.0)
    fit <- train(.y ~ ., data = train_df, method = method, trControl = tc, tuneGrid = tune,
                 metric = "Accuracy", scaled = FALSE)  # already standardized in Python
  } else {
    stop(paste("unknown learner", learner))
  }
}, error = function(e) {
  status <<- "failed"
  error <<- conditionMessage(e)
})

train_ms <- (proc.time()[[3]] - t_train0) * 1000

preds <- NULL
scores <- NULL
pred_ms <- NA_real_
acc <- NA_real_
if (identical(status, "ok") && !is.null(fit)) {
  t_pred0 <- proc.time()[[3]]
  pr <- predict(fit, newdata = data.frame(Xte, check.names = FALSE))
  preds <- from_class(pr)
  # probabilities / decision scores when available
  probs <- tryCatch(predict(fit, newdata = data.frame(Xte, check.names = FALSE), type = "prob"),
                    error = function(e) NULL)
  if (!is.null(probs) && "Class1" %in% colnames(probs)) {
    scores <- as.numeric(probs[["Class1"]])
  } else if (!is.null(probs) && ncol(probs) >= 2) {
    scores <- as.numeric(probs[[ncol(probs)]])
  }
  pred_ms <- (proc.time()[[3]] - t_pred0) * 1000
  acc <- mean(preds == yte)
}

adapter_ms <- prep_ms + ifelse(is.na(train_ms), 0, train_ms) + ifelse(is.na(pred_ms), 0, pred_ms)

result <- list(
  status = status,
  error = error,
  method = method,
  learner = learner,
  seed = seed,
  accuracy = acc,
  y_pred = preds,
  score = scores,
  y_true = yte,
  timings_ms = list(
    prepare_input_ms = prep_ms,
    train_ms = train_ms,
    predict_ms = pred_ms,
    adapter_wall_ms = adapter_ms,
    includes = c(
      "R process already started by caller",
      "package load not included (session already has caret)",
      "CSV already written by Python",
      "train + predict in this script"
    )
  )
)
write_json(result, out_path, auto_unbox = TRUE, pretty = TRUE, null = "null")
cat("caret", learner, status, "acc=", acc, "\n")
