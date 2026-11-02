-- Load pgml by absolute path to a built pgml.so (no CREATE EXTENSION).
-- Example:
--   psql -h /tmp -p 5433 -d ml_test -v SO=/usr/lib/postgresql/17/lib/pgml.so -f experiments/common/load_abs.sql
\if :{?SO}
\else
\echo 'Set SO to the absolute path of pgml.so, e.g. -v SO=/usr/lib/postgresql/17/lib/pgml.so'
\quit
\endif

-- ===== classification =====
CREATE OR REPLACE FUNCTION pgml_train_tree(float4[][], int[], int, int) RETURNS bytea AS :'SO', 'pgml_train_tree' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION pgml_predict_tree(bytea, float4[]) RETURNS int AS :'SO', 'pgml_predict_tree' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION pgml_predict_tree_batch(bytea, float4[][]) RETURNS int[] AS :'SO', 'pgml_predict_tree_batch' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION pgml_train_tree_fast(float4[][], int[], int, int) RETURNS bytea AS :'SO', 'pgml_train_tree_fast' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION pgml_train_forest(float4[][], int[], int, int, int) RETURNS bytea AS :'SO', 'pgml_train_forest' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION pgml_train_forest(float4[][], int[], int, int, int, int) RETURNS bytea AS :'SO', 'pgml_train_forest_seeded' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION pgml_predict_forest(bytea, float4[]) RETURNS int AS :'SO', 'pgml_predict_forest' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION pgml_predict_forest_batch(bytea, float4[][]) RETURNS int[] AS :'SO', 'pgml_predict_forest_batch' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION pgml_train_forest_fast(float4[][], int[], int, int, int) RETURNS bytea AS :'SO', 'pgml_train_forest_fast' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION pgml_train_forest_fast(float4[][], int[], int, int, int, int) RETURNS bytea AS :'SO', 'pgml_train_forest_fast_seeded' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION pgml_train_gbm(float4[][], int[], int, int, float4) RETURNS bytea AS :'SO', 'pgml_train_gbm' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION pgml_predict_gbm(bytea, float4[]) RETURNS int AS :'SO', 'pgml_predict_gbm' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION pgml_predict_gbm_batch(bytea, float4[][]) RETURNS int[] AS :'SO', 'pgml_predict_gbm_batch' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION pgml_train_gbm_fast(float4[][], int[], int, int, float4) RETURNS bytea AS :'SO', 'pgml_train_gbm_fast' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION pgml_train_adaboost(float4[][], int[], int) RETURNS bytea AS :'SO', 'pgml_train_adaboost' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION pgml_predict_adaboost(bytea, float4[]) RETURNS int AS :'SO', 'pgml_predict_adaboost' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION pgml_predict_adaboost_batch(bytea, float4[][]) RETURNS int[] AS :'SO', 'pgml_predict_adaboost_batch' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION pgml_train_svm(float4[][], int[], float4, int) RETURNS bytea AS :'SO', 'pgml_train_svm' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION pgml_predict_svm(bytea, float4[]) RETURNS int AS :'SO', 'pgml_predict_svm' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION pgml_predict_svm_batch(bytea, float4[][]) RETURNS int[] AS :'SO', 'pgml_predict_svm_batch' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION pgml_train_svm_ovr(float4[][], int[], int, float4, int) RETURNS bytea AS :'SO', 'pgml_train_svm_ovr' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION pgml_predict_svm_ovr_batch(bytea, float4[][]) RETURNS int[] AS :'SO', 'pgml_predict_svm_ovr_batch' LANGUAGE C STRICT;

-- ===== clustering =====
CREATE OR REPLACE FUNCTION pgml_kmeans(float4[][], int, int, int) RETURNS bytea AS :'SO', 'pgclust_kmeans' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION pgml_kmeans_random(float4[][], int, int, int) RETURNS bytea AS :'SO', 'pgclust_kmeans_random' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION pgml_kmeans_predict_batch(bytea, float4[][]) RETURNS int[] AS :'SO', 'pgclust_kmeans_predict_batch' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION pgml_dbscan(float4[][], float4, int) RETURNS int[] AS :'SO', 'pgclust_dbscan' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION pgml_dbscan_fast(float4[][], float4, int) RETURNS int[] AS :'SO', 'pgclust_dbscan_fast' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION pgml_hierarchical(float4[][], int, int) RETURNS int[] AS :'SO', 'pgclust_hierarchical' LANGUAGE C STRICT;

-- ===== regression =====
CREATE OR REPLACE FUNCTION pgml_linreg(float4[][], float4[]) RETURNS bytea AS :'SO', 'pgreg_linreg' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION pgml_ridge(float4[][], float4[], float4) RETURNS bytea AS :'SO', 'pgreg_ridge' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION pgml_predict_linreg_batch(bytea, float4[][]) RETURNS float4[] AS :'SO', 'pgreg_predict_linreg_batch' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION pgml_regression_tree(float4[][], float4[], int, int) RETURNS bytea AS :'SO', 'pgreg_tree' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION pgml_predict_regression_tree_batch(bytea, float4[][]) RETURNS float4[] AS :'SO', 'pgreg_predict_tree_batch' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION pgml_gbm_regression(float4[][], float4[], int, int, float4) RETURNS bytea AS :'SO', 'pgreg_gbm' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION pgml_predict_gbm_regression_batch(bytea, float4[][]) RETURNS float4[] AS :'SO', 'pgreg_predict_gbm_batch' LANGUAGE C STRICT;

-- ===== PCA =====
CREATE OR REPLACE FUNCTION pgml_pca_fit(float4[][], int) RETURNS bytea AS :'SO', 'pgpca_fit' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION pgml_pca_transform(bytea, float4[][]) RETURNS float4[][] AS :'SO', 'pgpca_transform' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION pgml_pca_eigenvalues(bytea) RETURNS float4[] AS :'SO', 'pgpca_eigenvalues' LANGUAGE C STRICT;

-- ===== multithreaded variants =====
CREATE OR REPLACE FUNCTION pgml_kmeans_parallel(float4[][], int, int, int, int) RETURNS bytea AS :'SO', 'pgclust_kmeans_parallel' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION pgml_pca_fit_parallel(float4[][], int, int) RETURNS bytea AS :'SO', 'pgpca_fit_parallel' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION pgml_linreg_parallel(float4[][], float4[], int) RETURNS bytea AS :'SO', 'pgreg_linreg_parallel' LANGUAGE C STRICT;

-- ===== categorical features =====
CREATE OR REPLACE FUNCTION pgml_train_tree_cat(float4[][], int[], int, int, int[]) RETURNS bytea AS :'SO', 'pgml_train_tree_cat' LANGUAGE C STRICT;
