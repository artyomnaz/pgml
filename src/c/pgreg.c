/* pgreg.c — In-database regression: linear (OLS), ridge, CART regression tree,
 * gradient-boosting regression. Native C for PostgreSQL, no external libraries.
 */
#include "postgres.h"
#include "fmgr.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include "utils/memutils.h"
#include "pgml_timing.h"
#include <pthread.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

PG_MODULE_MAGIC;

/* ====================== helpers ====================== */

static float4 *get_matrix(ArrayType *arr, int *nrows, int *ncols)
{
    if (ARR_NDIM(arr) != 2) elog(ERROR, "pgreg: features must be a 2-D array");
    *nrows = ARR_DIMS(arr)[0]; *ncols = ARR_DIMS(arr)[1];
    Datum *e; bool *nulls; int ne;
    deconstruct_array(arr, FLOAT4OID, sizeof(float4), true, 'i', &e, &nulls, &ne);
    if (ne != (*nrows) * (*ncols)) elog(ERROR, "pgreg: ragged feature matrix");
    float4 *X = palloc(sizeof(float4) * ne);
    for (int i = 0; i < ne; i++) X[i] = nulls[i] ? 0.0f : DatumGetFloat4(e[i]);
    return X;
}

static float4 *get_vector(ArrayType *arr, int *n)
{
    Datum *e; bool *nulls; int ne;
    deconstruct_array(arr, FLOAT4OID, sizeof(float4), true, 'i', &e, &nulls, &ne);
    float4 *v = palloc(sizeof(float4) * ne);
    for (int i = 0; i < ne; i++) v[i] = nulls[i] ? 0.0f : DatumGetFloat4(e[i]);
    *n = ne;
    return v;
}

static bytea *make_bytea(const void *data, int len)
{
    bytea *b = palloc(VARHDRSZ + len);
    SET_VARSIZE(b, VARHDRSZ + len);
    if (len > 0) memcpy(VARDATA(b), data, len);
    return b;
}

static ArrayType *float_array_from(float *vals, int n)
{
    Datum *out = palloc(sizeof(Datum) * n);
    for (int i = 0; i < n; i++) out[i] = Float4GetDatum(vals[i]);
    return construct_array(out, n, FLOAT4OID, sizeof(float4), true, 'i');
}

/* Gauss elimination with partial pivoting (A is n x n, modified; b is n) */
static void solve_linear(double *A, double *b, int n, double *x)
{
    for (int col = 0; col < n; col++) {
        int piv = col;
        for (int r = col + 1; r < n; r++) if (fabs(A[r*n+col]) > fabs(A[piv*n+col])) piv = r;
        if (fabs(A[piv*n+col]) < 1e-12) continue;
        if (piv != col) {
            for (int k = 0; k < n; k++) { double t = A[col*n+k]; A[col*n+k] = A[piv*n+k]; A[piv*n+k] = t; }
            double t = b[col]; b[col] = b[piv]; b[piv] = t;
        }
        for (int r = col + 1; r < n; r++) {
            double f = A[r*n+col] / A[col*n+col];
            A[r*n+col] = 0;
            for (int k = col + 1; k < n; k++) A[r*n+k] -= f * A[col*n+k];
            b[r] -= f * b[col];
        }
    }
    for (int r = n - 1; r >= 0; r--) {
        double s = b[r];
        for (int k = r + 1; k < n; k++) s -= A[r*n+k] * x[k];
        x[r] = (fabs(A[r*n+r]) > 1e-12) ? s / A[r*n+r] : 0.0;
    }
}

/* ====================== linear / ridge regression ====================== */

typedef struct LinReg { int p; float *w; } LinReg;   /* p = d+1, w[0]=bias */

static LinReg train_linreg(float4 *X, float4 *y, int n, int d, float lambda)
{
    int p = d + 1;
    double *A = palloc0(sizeof(double) * p * p);
    double *b = palloc0(sizeof(double) * p);
    for (int i = 0; i < n; i++) {
        double yi = y[i];
        for (int j = 0; j < p; j++) {
            double xij = (j == 0) ? 1.0 : X[i*d + (j-1)];
            b[j] += xij * yi;
            for (int k = 0; k < p; k++) {
                double xik = (k == 0) ? 1.0 : X[i*d + (k-1)];
                A[j*p + k] += xij * xik;
            }
        }
    }
    for (int j = 0; j < p; j++) A[j*p + j] += lambda;
    double *wsol = palloc0(sizeof(double) * p);
    solve_linear(A, b, p, wsol);
    LinReg m; m.p = p; m.w = palloc(sizeof(float) * p);
    for (int j = 0; j < p; j++) m.w[j] = (float)wsol[j];
    pfree(A); pfree(b); pfree(wsol);
    return m;
}

static double predict_linreg(const LinReg *m, const float *x)
{
    double z = m->w[0];
    for (int j = 0; j < m->p - 1; j++) z += m->w[j+1] * x[j];
    return z;
}

static bytea *ser_linreg(const LinReg *m)
{
    int len = sizeof(int) + m->p * sizeof(float);
    char *buf = palloc(len), *p = buf;
    memcpy(p, &m->p, sizeof(int)); p += sizeof(int);
    memcpy(p, m->w, m->p * sizeof(float));
    return make_bytea(buf, len);
}

static LinReg deser_linreg(const bytea *b)
{
    const char *p = VARDATA(b);
    LinReg m; memcpy(&m.p, p, sizeof(int)); p += sizeof(int);
    m.w = palloc(sizeof(float) * m.p);
    memcpy(m.w, p, m.p * sizeof(float));
    return m;
}

/* ====================== regression tree (CART, squared error) ====================== */

typedef struct TNode { int feature; float threshold; int left, right; float value; } TNode;
typedef struct Tree { int nnodes; TNode *nodes; } Tree;
typedef struct { int idx; float val; } SortPair;

static int cmp_sp(const void *a, const void *b) {
    const SortPair *x = a, *y = b;
    return (x->val > y->val) - (x->val < y->val);
}

static int build_rtree(float4 *X, double *r, int n, int d, int *idx,
                       int depth, int max_depth, TNode *nodes, int *nnodes, int cap)
{
    int me = (*nnodes)++;
    nodes[me].feature = -1; nodes[me].threshold = 0;
    nodes[me].left = nodes[me].right = -1;
    double mean = 0; for (int i = 0; i < n; i++) mean += r[idx[i]];
    mean /= n; nodes[me].value = (float)mean;
    if (depth >= max_depth || n < 4) return me;
    double tv = 0; for (int i = 0; i < n; i++) { double e = r[idx[i]] - mean; tv += e * e; }
    if (tv <= 1e-12) return me;
    int bf = -1; float bt = 0; double bg = 0;
    SortPair *sp = palloc(sizeof(SortPair) * n);
    for (int f = 0; f < d; f++) {
        for (int i = 0; i < n; i++) { sp[i].idx = idx[i]; sp[i].val = X[idx[i]*d+f]; }
        qsort(sp, n, sizeof(SortPair), cmp_sp);
        double sl = 0, sr = 0; for (int i = 0; i < n; i++) sr += r[idx[i]];
        for (int i = 0; i < n - 1; i++) {
            sl += r[sp[i].idx]; sr -= r[sp[i].idx];
            if (sp[i].val == sp[i+1].val) continue;
            float t = 0.5f * (sp[i].val + sp[i+1].val);
            int nl = i + 1, nr = n - nl;
            double ml = sl / nl, mr = sr / nr;
            double vl = 0, vr = 0;
            for (int j = 0; j <= i; j++) { double e = r[sp[j].idx] - ml; vl += e * e; }
            for (int j = i + 1; j < n; j++) { double e = r[sp[j].idx] - mr; vr += e * e; }
            double gain = tv - vl - vr;
            if (gain > bg) { bg = gain; bf = f; bt = t; }
        }
    }
    pfree(sp);
    if (bf < 0 || bg <= 1e-12) return me;
    int *li = palloc(sizeof(int)*n), *ri = palloc(sizeof(int)*n);
    int nl = 0, nr = 0;
    for (int i = 0; i < n; i++)
        if (X[idx[i]*d + bf] <= bt) li[nl++] = idx[i]; else ri[nr++] = idx[i];
    if (nl == 0 || nr == 0) { pfree(li); pfree(ri); return me; }
    nodes[me].feature = bf; nodes[me].threshold = bt;
    nodes[me].left  = build_rtree(X, r, nl, d, li, depth+1, max_depth, nodes, nnodes, cap);
    nodes[me].right = build_rtree(X, r, nr, d, ri, depth+1, max_depth, nodes, nnodes, cap);
    pfree(li); pfree(ri);
    return me;
}

static float predict_rtree(const Tree *t, const float *x)
{
    int node = 0;
    while (t->nodes[node].feature >= 0)
        node = (x[t->nodes[node].feature] <= t->nodes[node].threshold) ? t->nodes[node].left : t->nodes[node].right;
    return t->nodes[node].value;
}

#define NODE_REC (sizeof(int)*3 + sizeof(float)*2)
static void write_tree(char **p, const Tree *t)
{
    memcpy(*p, &t->nnodes, sizeof(int)); *p += sizeof(int);
    for (int i = 0; i < t->nnodes; i++) {
        TNode *nd = &t->nodes[i];
        memcpy(*p, &nd->feature, sizeof(int)); *p += sizeof(int);
        memcpy(*p, &nd->threshold, sizeof(float)); *p += sizeof(float);
        memcpy(*p, &nd->left, sizeof(int)); *p += sizeof(int);
        memcpy(*p, &nd->right, sizeof(int)); *p += sizeof(int);
        memcpy(*p, &nd->value, sizeof(float)); *p += sizeof(float);
    }
}
static void read_tree(const char **p, Tree *t)
{
    memcpy(&t->nnodes, *p, sizeof(int)); *p += sizeof(int);
    t->nodes = palloc(sizeof(TNode) * t->nnodes);
    for (int i = 0; i < t->nnodes; i++) {
        TNode *nd = &t->nodes[i];
        memcpy(&nd->feature, *p, sizeof(int)); *p += sizeof(int);
        memcpy(&nd->threshold, *p, sizeof(float)); *p += sizeof(float);
        memcpy(&nd->left, *p, sizeof(int)); *p += sizeof(int);
        memcpy(&nd->right, *p, sizeof(int)); *p += sizeof(int);
        memcpy(&nd->value, *p, sizeof(float)); *p += sizeof(float);
    }
}

/* ====================== gradient-boosting regression ====================== */

typedef struct GBMReg { int nrounds; float lr; float base; Tree *trees; } GBMReg;

static GBMReg train_gbm_reg(float4 *X, float4 *y, int n, int d, int nrounds, int depth, float lr)
{
    GBMReg g; g.nrounds = nrounds; g.lr = lr; g.trees = palloc(sizeof(Tree) * nrounds);
    double *pred = palloc(sizeof(double) * n);
    double *resid = palloc(sizeof(double) * n);
    int *idx = palloc(sizeof(int) * n);
    double mean = 0; for (int i = 0; i < n; i++) mean += y[i]; mean /= n;
    g.base = (float)mean;
    for (int i = 0; i < n; i++) { pred[i] = mean; idx[i] = i; }
    for (int r = 0; r < nrounds; r++) {
        for (int i = 0; i < n; i++) resid[i] = (double)y[i] - pred[i];
        int cap = 2 * n + 16;
        Tree t; t.nodes = palloc(sizeof(TNode) * cap); t.nnodes = 0;
        build_rtree(X, resid, n, d, idx, 0, depth, t.nodes, &t.nnodes, cap);
        for (int i = 0; i < n; i++) pred[i] += lr * predict_rtree(&t, X + i*d);
        g.trees[r] = t;
    }
    pfree(pred); pfree(resid); pfree(idx);
    return g;
}

static double predict_gbm_reg(const GBMReg *g, const float *x)
{
    double z = g->base;
    for (int r = 0; r < g->nrounds; r++) z += g->lr * predict_rtree(&g->trees[r], x);
    return z;
}

static bytea *ser_gbm_reg(const GBMReg *g)
{
    int len = sizeof(int) + 2 * sizeof(float);
    for (int r = 0; r < g->nrounds; r++) len += sizeof(int) + g->trees[r].nnodes * NODE_REC;
    char *buf = palloc(len), *p = buf;
    memcpy(p, &g->nrounds, sizeof(int)); p += sizeof(int);
    memcpy(p, &g->lr, sizeof(float)); p += sizeof(float);
    memcpy(p, &g->base, sizeof(float)); p += sizeof(float);
    for (int r = 0; r < g->nrounds; r++) write_tree(&p, &g->trees[r]);
    return make_bytea(buf, len);
}
static GBMReg deser_gbm_reg(const bytea *b)
{
    const char *p = VARDATA(b);
    GBMReg g; memcpy(&g.nrounds, p, sizeof(int)); p += sizeof(int);
    memcpy(&g.lr, p, sizeof(float)); p += sizeof(float);
    memcpy(&g.base, p, sizeof(float)); p += sizeof(float);
    g.trees = palloc(sizeof(Tree) * g.nrounds);
    for (int i = 0; i < g.nrounds; i++) read_tree(&p, &g.trees[i]);
    return g;
}

/* ====================== PG wrappers ====================== */

PG_FUNCTION_INFO_V1(pgreg_linreg);
Datum pgreg_linreg(PG_FUNCTION_ARGS)
{
    PgmlStageTimer st; pgml_stage_begin(&st);
    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType *lab = PG_GETARG_ARRAYTYPE_P(1);
    int n, d; float4 *X = get_matrix(arr, &n, &d); float4 *y = get_vector(lab, &n);
    st.convert_ms = pgml_stage_lap_ms(&st);
    LinReg m = train_linreg(X, y, n, d, 0.0f);
    st.native_ms = pgml_stage_lap_ms(&st);
    bytea *res = ser_linreg(&m);
    st.serialize_ms = pgml_stage_lap_ms(&st);
    pgml_stage_emit(&st);
    PG_RETURN_BYTEA_P(res);
}

PG_FUNCTION_INFO_V1(pgreg_ridge);
Datum pgreg_ridge(PG_FUNCTION_ARGS)
{
    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType *lab = PG_GETARG_ARRAYTYPE_P(1);
    float lambda = PG_GETARG_FLOAT4(2);
    int n, d; float4 *X = get_matrix(arr, &n, &d); float4 *y = get_vector(lab, &n);
    LinReg m = train_linreg(X, y, n, d, lambda);
    PG_RETURN_BYTEA_P(ser_linreg(&m));
}

PG_FUNCTION_INFO_V1(pgreg_predict_linreg_batch);
Datum pgreg_predict_linreg_batch(PG_FUNCTION_ARGS)
{
    bytea *b = PG_GETARG_BYTEA_P(0);
    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(1);
    int n, d; float4 *X = get_matrix(arr, &n, &d);
    LinReg m = deser_linreg(b);
    float *out = palloc(sizeof(float) * n);
    for (int i = 0; i < n; i++) out[i] = (float)predict_linreg(&m, X + i*d);
    PG_RETURN_ARRAYTYPE_P(float_array_from(out, n));
}

PG_FUNCTION_INFO_V1(pgreg_tree);
Datum pgreg_tree(PG_FUNCTION_ARGS)
{
    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType *lab = PG_GETARG_ARRAYTYPE_P(1);
    int max_depth = PG_GETARG_INT32(2);
    int min_samples = PG_GETARG_INT32(3);
    int n, d; float4 *X = get_matrix(arr, &n, &d); float4 *y = get_vector(lab, &n);
    int *idx = palloc(sizeof(int) * n); for (int i = 0; i < n; i++) idx[i] = i;
    double *r = palloc(sizeof(double) * n); for (int i = 0; i < n; i++) r[i] = y[i];
    int cap = 2 * n + 16;
    Tree t; t.nodes = palloc(sizeof(TNode) * cap); t.nnodes = 0;
    build_rtree(X, r, n, d, idx, 0, max_depth, t.nodes, &t.nnodes, cap);
    int len = sizeof(int) + t.nnodes * NODE_REC;
    char *buf = palloc(len), *p = buf; write_tree(&p, &t);
    bytea *res = make_bytea(buf, len);
    pfree(idx); pfree(r); pfree(buf);
    PG_RETURN_BYTEA_P(res);
}

PG_FUNCTION_INFO_V1(pgreg_predict_tree_batch);
Datum pgreg_predict_tree_batch(PG_FUNCTION_ARGS)
{
    bytea *b = PG_GETARG_BYTEA_P(0);
    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(1);
    int n, d; float4 *X = get_matrix(arr, &n, &d);
    Tree t; const char *p = VARDATA(b); read_tree(&p, &t);
    float *out = palloc(sizeof(float) * n);
    for (int i = 0; i < n; i++) out[i] = predict_rtree(&t, X + i*d);
    PG_RETURN_ARRAYTYPE_P(float_array_from(out, n));
}

PG_FUNCTION_INFO_V1(pgreg_gbm);
Datum pgreg_gbm(PG_FUNCTION_ARGS)
{
    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType *lab = PG_GETARG_ARRAYTYPE_P(1);
    int nrounds = PG_GETARG_INT32(2);
    int depth = PG_GETARG_INT32(3);
    float lr = PG_GETARG_FLOAT4(4);
    int n, d; float4 *X = get_matrix(arr, &n, &d); float4 *y = get_vector(lab, &n);
    GBMReg g = train_gbm_reg(X, y, n, d, nrounds, depth, lr);
    PG_RETURN_BYTEA_P(ser_gbm_reg(&g));
}

PG_FUNCTION_INFO_V1(pgreg_predict_gbm_batch);
Datum pgreg_predict_gbm_batch(PG_FUNCTION_ARGS)
{
    bytea *b = PG_GETARG_BYTEA_P(0);
    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(1);
    int n, d; float4 *X = get_matrix(arr, &n, &d);
    GBMReg g = deser_gbm_reg(b);
    float *out = palloc(sizeof(float) * n);
    for (int i = 0; i < n; i++) out[i] = (float)predict_gbm_reg(&g, X + i*d);
    PG_RETURN_ARRAYTYPE_P(float_array_from(out, n));
}

/* ====================== multithreaded linear regression (parallel normal equations) ====================== */

typedef struct { const float *X; const float *y; int n, d, p, start, end; double *A; double *b; } LRJob;

static void *lr_worker(void *arg)
{
    LRJob *j = arg;
    for (int i = j->start; i < j->end; i++) {
        double yi = j->y[i];
        for (int k = 0; k < j->p; k++) {
            double xik = (k == 0) ? 1.0 : j->X[i*j->d + (k-1)];
            j->b[k] += xik * yi;
            for (int l = 0; l < j->p; l++) {
                double xil = (l == 0) ? 1.0 : j->X[i*j->d + (l-1)];
                j->A[k*j->p + l] += xik * xil;
            }
        }
    }
    return NULL;
}

static LinReg train_linreg_parallel(float4 *X, float4 *y, int n, int d, float lambda, int n_threads)
{
    int p = d + 1;
    if (n_threads < 1) n_threads = 1;
    if (n_threads > n) n_threads = n;
    pthread_t *th = malloc(sizeof(pthread_t) * n_threads);
    LRJob *jobs = malloc(sizeof(LRJob) * n_threads);
    double **As = malloc(sizeof(double *) * n_threads);
    double **bs = malloc(sizeof(double *) * n_threads);
    for (int t = 0; t < n_threads; t++) { As[t] = calloc(p * p, sizeof(double)); bs[t] = calloc(p, sizeof(double)); }
    int chunk = (n + n_threads - 1) / n_threads;
    for (int t = 0; t < n_threads; t++) {
        jobs[t].X = X; jobs[t].y = y; jobs[t].n = n; jobs[t].d = d; jobs[t].p = p;
        jobs[t].start = t * chunk;
        jobs[t].end = ((t + 1) * chunk < n) ? (t + 1) * chunk : n;
        jobs[t].A = As[t]; jobs[t].b = bs[t];
        pthread_create(&th[t], NULL, lr_worker, &jobs[t]);
    }
    for (int t = 0; t < n_threads; t++) pthread_join(th[t], NULL);
    double *A = palloc0(sizeof(double) * p * p);
    double *b = palloc0(sizeof(double) * p);
    for (int t = 0; t < n_threads; t++) {
        for (int i = 0; i < p * p; i++) A[i] += As[t][i];
        for (int i = 0; i < p; i++) b[i] += bs[t][i];
    }
    for (int t = 0; t < n_threads; t++) { free(As[t]); free(bs[t]); }
    free(As); free(bs); free(jobs); free(th);
    for (int j = 0; j < p; j++) A[j*p + j] += lambda;
    double *wsol = palloc0(sizeof(double) * p);
    solve_linear(A, b, p, wsol);
    LinReg m; m.p = p; m.w = palloc(sizeof(float) * p);
    for (int j = 0; j < p; j++) m.w[j] = (float)wsol[j];
    pfree(A); pfree(b); pfree(wsol);
    return m;
}

PG_FUNCTION_INFO_V1(pgreg_linreg_parallel);
Datum pgreg_linreg_parallel(PG_FUNCTION_ARGS)
{
    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType *lab = PG_GETARG_ARRAYTYPE_P(1);
    int n_threads = PG_GETARG_INT32(2);
    int n, d; float4 *X = get_matrix(arr, &n, &d); float4 *y = get_vector(lab, &n);
    LinReg m = train_linreg_parallel(X, y, n, d, 0.0f, n_threads);
    PG_RETURN_BYTEA_P(ser_linreg(&m));
}
