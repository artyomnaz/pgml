/* pgml.c — In-database ML: decision tree, random forest, gradient boosting,
 * AdaBoost, linear SVM. Native C for PostgreSQL (no external ML libraries).
 * Training inputs: float4[][] features, int[] labels. Models serialize to bytea.
 */
#include "postgres.h"
#include "fmgr.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include "utils/memutils.h"
#include "pgml_timing.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

PG_MODULE_MAGIC;

/* ====================== helpers ====================== */

static float4 *get_matrix(ArrayType *arr, int *nrows, int *ncols)
{
    if (ARR_NDIM(arr) != 2)
        elog(ERROR, "pgml: features must be a 2-D array (nrows x ncols)");
    *nrows = ARR_DIMS(arr)[0];
    *ncols = ARR_DIMS(arr)[1];
    Datum *elems; bool *nulls; int nelems;
    deconstruct_array(arr, FLOAT4OID, sizeof(float4), true, 'i', &elems, &nulls, &nelems);
    if (nelems != (*nrows) * (*ncols))
        elog(ERROR, "pgml: ragged feature matrix");
    float4 *X = palloc(sizeof(float4) * nelems);
    for (int i = 0; i < nelems; i++)
        X[i] = nulls[i] ? 0.0f : DatumGetFloat4(elems[i]);
    return X;
}

static int32 *get_labels(ArrayType *arr, int *n)
{
    Datum *e; bool *nulls; int ne;
    deconstruct_array(arr, INT4OID, sizeof(int32), true, 'i', &e, &nulls, &ne);
    int32 *y = palloc(sizeof(int32) * ne);
    for (int i = 0; i < ne; i++) y[i] = nulls[i] ? 0 : DatumGetInt32(e[i]);
    *n = ne;
    return y;
}

static float4 *get_vector(ArrayType *arr, int *d)
{
    Datum *e; bool *nulls; int ne;
    deconstruct_array(arr, FLOAT4OID, sizeof(float4), true, 'i', &e, &nulls, &ne);
    float4 *v = palloc(sizeof(float4) * ne);
    for (int i = 0; i < ne; i++) v[i] = nulls[i] ? 0.0f : DatumGetFloat4(e[i]);
    *d = ne;
    return v;
}

static bytea *make_bytea(const void *data, int len)
{
    bytea *b = palloc(VARHDRSZ + len);
    SET_VARSIZE(b, VARHDRSZ + len);
    if (len > 0) memcpy(VARDATA(b), data, len);
    return b;
}

/* ====================== tree (shared node struct) ====================== */

typedef struct TNode {
    int    feature;   /* -1 => leaf */
    float  threshold;
    int    left, right;
    int    label;     /* classification leaf class */
    float  value;     /* regression leaf prediction */
} TNode;

typedef struct Tree { int nnodes; TNode *nodes; } Tree;

typedef struct { int idx; float val; } SortPair;
static int cmp_sortpair(const void *a, const void *b)
{
    const SortPair *x = a, *y = b;
    return (x->val > y->val) - (x->val < y->val);
}

/* ---------- classification tree (CART, Gini) ---------- */

static int majority_class(int32 *y, int *idx, int n)
{
    int counts[64] = {0};
    for (int i = 0; i < n; i++) { int c = y[idx[i]]; if (c >= 0 && c < 64) counts[c]++; }
    int best = 0, bestc = -1;
    for (int c = 0; c < 64; c++) if (counts[c] > bestc) { bestc = counts[c]; best = c; }
    return best;
}

static double cls_gini(int32 *y, int *idx, int n, int *counts)
{
    memset(counts, 0, sizeof(int) * 64);
    for (int i = 0; i < n; i++) { int c = y[idx[i]]; if (c >= 0 && c < 64) counts[c]++; }
    double g = 1.0;
    for (int c = 0; c < 64; c++) { double p = (double)counts[c] / n; g -= p * p; }
    return g;
}

static int build_tree(float4 *X, int32 *y, int n, int d, int *idx,
                      int depth, int max_depth, int min_samples,
                      TNode *nodes, int *nnodes, int cap)
{
    int me = (*nnodes)++;
    if (me >= cap) elog(ERROR, "pgml: tree capacity exceeded");
    nodes[me].feature = -1; nodes[me].threshold = 0;
    nodes[me].left = nodes[me].right = -1;
    nodes[me].value = 0;
    nodes[me].label = majority_class(y, idx, n);

    int base[64];
    double parent_g = cls_gini(y, idx, n, base);
    if (parent_g <= 1e-9 || depth >= max_depth || n < min_samples) return me;

    int best_f = -1; float best_t = 0; double best_gain = 0;
    SortPair *sp = palloc(sizeof(SortPair) * n);

    for (int f = 0; f < d; f++) {
        for (int i = 0; i < n; i++) { sp[i].idx = idx[i]; sp[i].val = X[idx[i]*d + f]; }
        qsort(sp, n, sizeof(SortPair), cmp_sortpair);
        int left_c[64] = {0}, right_c[64] = {0};
        for (int i = 0; i < n; i++) { int c = y[idx[i]]; if (c >= 0 && c < 64) right_c[c]++; }
        for (int i = 0; i < n - 1; i++) {
            int c = y[sp[i].idx]; if (c >= 0 && c < 64) { left_c[c]++; right_c[c]--; }
            if (sp[i].val == sp[i+1].val) continue;
            float t = 0.5f * (sp[i].val + sp[i+1].val);
            int nl = i + 1, nr = n - nl;
            double gl = 1.0, gr = 1.0;
            for (int cc = 0; cc < 64; cc++) {
                double pl = (double)left_c[cc] / nl;  gl -= pl * pl;
                double pr = (double)right_c[cc] / nr; gr -= pr * pr;
            }
            double gain = parent_g - ((double)nl/n)*gl - ((double)nr/n)*gr;
            if (gain > best_gain) { best_gain = gain; best_f = f; best_t = t; }
        }
    }
    pfree(sp);
    if (best_f < 0 || best_gain <= 1e-9) return me;

    int *li = palloc(sizeof(int)*n), *ri = palloc(sizeof(int)*n);
    int nl = 0, nr = 0;
    for (int i = 0; i < n; i++) {
        if (X[idx[i]*d + best_f] <= best_t) li[nl++] = idx[i]; else ri[nr++] = idx[i];
    }
    if (nl == 0 || nr == 0) { pfree(li); pfree(ri); return me; }
    nodes[me].feature = best_f; nodes[me].threshold = best_t;
    nodes[me].left  = build_tree(X, y, nl, d, li, depth+1, max_depth, min_samples, nodes, nnodes, cap);
    nodes[me].right = build_tree(X, y, nr, d, ri, depth+1, max_depth, min_samples, nodes, nnodes, cap);
    pfree(li); pfree(ri);
    return me;
}

static Tree train_tree_idx(float4 *X, int32 *y, int n, int d, int *idx,
                           int max_depth, int min_samples)
{
    int cap = 2 * n + 16;
    Tree t; t.nodes = palloc(sizeof(TNode) * cap); t.nnodes = 0;
    build_tree(X, y, n, d, idx, 0, max_depth, min_samples, t.nodes, &t.nnodes, cap);
    return t;
}

static Tree train_tree(float4 *X, int32 *y, int n, int d, int max_depth, int min_samples)
{
    int *idx = palloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) idx[i] = i;
    Tree t = train_tree_idx(X, y, n, d, idx, max_depth, min_samples);
    pfree(idx);
    return t;
}

static int predict_cls_tree(const Tree *t, const float4 *x)
{
    int node = 0;
    while (t->nodes[node].feature != -1) {
        int f = t->nodes[node].feature;
        bool go_left;
        if (f >= 0)      go_left = (x[f] <= t->nodes[node].threshold);
        else             go_left = (x[-f-2] == t->nodes[node].threshold);  /* categorical membership */
        node = go_left ? t->nodes[node].left : t->nodes[node].right;
    }
    return t->nodes[node].label;
}

/* ---------- regression tree (squared error) for GBM ---------- */

static int build_rtree(float4 *X, double *r, int n, int d, int *idx,
                       int depth, int max_depth,
                       TNode *nodes, int *nnodes, int cap)
{
    int me = (*nnodes)++;
    nodes[me].feature = -1; nodes[me].threshold = 0;
    nodes[me].left = nodes[me].right = -1; nodes[me].label = 0;

    double mean = 0; for (int i = 0; i < n; i++) mean += r[idx[i]];
    mean /= n;
    nodes[me].value = (float)mean;

    if (depth >= max_depth || n < 4) return me;

    double total_var = 0;
    for (int i = 0; i < n; i++) { double e = r[idx[i]] - mean; total_var += e * e; }
    if (total_var <= 1e-12) return me;

    int best_f = -1; float best_t = 0; double best_gain = 0;
    SortPair *sp = palloc(sizeof(SortPair) * n);
    for (int f = 0; f < d; f++) {
        for (int i = 0; i < n; i++) { sp[i].idx = idx[i]; sp[i].val = X[idx[i]*d + f]; }
        qsort(sp, n, sizeof(SortPair), cmp_sortpair);
        double sum_l = 0, sum_r = 0;
        for (int i = 0; i < n; i++) sum_r += r[idx[i]];
        for (int i = 0; i < n - 1; i++) {
            sum_l += r[sp[i].idx]; sum_r -= r[sp[i].idx];
            if (sp[i].val == sp[i+1].val) continue;
            float t = 0.5f * (sp[i].val + sp[i+1].val);
            int nl = i + 1, nr = n - nl;
            double ml = sum_l / nl, mr = sum_r / nr;
            double vl = 0, vr = 0;
            for (int j = 0; j <= i; j++) { double e = r[sp[j].idx] - ml; vl += e * e; }
            for (int j = i + 1; j < n; j++) { double e = r[sp[j].idx] - mr; vr += e * e; }
            double gain = total_var - (vl + vr);
            if (gain > best_gain) { best_gain = gain; best_f = f; best_t = t; }
        }
    }
    pfree(sp);
    if (best_f < 0 || best_gain <= 1e-12) return me;

    int *li = palloc(sizeof(int)*n), *ri = palloc(sizeof(int)*n);
    int nl = 0, nr = 0;
    for (int i = 0; i < n; i++)
        if (X[idx[i]*d + best_f] <= best_t) li[nl++] = idx[i]; else ri[nr++] = idx[i];
    if (nl == 0 || nr == 0) { pfree(li); pfree(ri); return me; }
    nodes[me].feature = best_f; nodes[me].threshold = best_t;
    nodes[me].left  = build_rtree(X, r, nl, d, li, depth+1, max_depth, nodes, nnodes, cap);
    nodes[me].right = build_rtree(X, r, nr, d, ri, depth+1, max_depth, nodes, nnodes, cap);
    pfree(li); pfree(ri);
    return me;
}

static float predict_reg_tree(const Tree *t, const float4 *x)
{
    int node = 0;
    while (t->nodes[node].feature >= 0)
        node = (x[t->nodes[node].feature] <= t->nodes[node].threshold)
               ? t->nodes[node].left : t->nodes[node].right;
    return t->nodes[node].value;
}

/* ---------- tree (de)serialization (uniform node record) ---------- */

#define NODE_REC (sizeof(int)*4 + sizeof(float)*2)

static void put_node(char **p, const TNode *nd)
{
    memcpy(*p, &nd->feature, sizeof(int)); *p += sizeof(int);
    memcpy(*p, &nd->threshold, sizeof(float)); *p += sizeof(float);
    memcpy(*p, &nd->left, sizeof(int)); *p += sizeof(int);
    memcpy(*p, &nd->right, sizeof(int)); *p += sizeof(int);
    memcpy(*p, &nd->label, sizeof(int)); *p += sizeof(int);
    memcpy(*p, &nd->value, sizeof(float)); *p += sizeof(float);
}
static void get_node(const char **p, TNode *nd)
{
    memcpy(&nd->feature, *p, sizeof(int)); *p += sizeof(int);
    memcpy(&nd->threshold, *p, sizeof(float)); *p += sizeof(float);
    memcpy(&nd->left, *p, sizeof(int)); *p += sizeof(int);
    memcpy(&nd->right, *p, sizeof(int)); *p += sizeof(int);
    memcpy(&nd->label, *p, sizeof(int)); *p += sizeof(int);
    memcpy(&nd->value, *p, sizeof(float)); *p += sizeof(float);
}

static void write_tree(char **p, const Tree *t)
{
    memcpy(*p, &t->nnodes, sizeof(int)); *p += sizeof(int);
    for (int i = 0; i < t->nnodes; i++) put_node(p, &t->nodes[i]);
}
static void read_tree(const char **p, Tree *t)
{
    memcpy(&t->nnodes, *p, sizeof(int)); *p += sizeof(int);
    t->nodes = palloc(sizeof(TNode) * t->nnodes);
    for (int i = 0; i < t->nnodes; i++) get_node(p, &t->nodes[i]);
}

/* ====================== random forest ====================== */

typedef struct Forest { int ntrees; Tree *trees; } Forest;

/* Local LCG for bootstrap sampling. Stack-allocated per call so training
 * does not depend on / mutate process-global rand()/srand() state. Forest
 * training is single-threaded; no shared RNG across threads. */
typedef struct { uint32 state; } ForestRNG;
static void forest_rng_seed(ForestRNG *rng, uint32 seed)
{
    rng->state = seed ? seed : 1u;
}
static uint32 forest_rng_next(ForestRNG *rng)
{
    rng->state = rng->state * 1664525u + 1013904223u;
    return rng->state;
}
/* Uniform index in [0, n). Uses [0,1) from 2^32 to avoid RAND_MAX edge. */
static int forest_rng_index(ForestRNG *rng, int n)
{
    if (n <= 0)
        return 0;
    return (int)((forest_rng_next(rng) / 4294967296.0) * n);
}

#define PGML_FOREST_DEFAULT_SEED 1u

static Forest train_forest(float4 *X, int32 *y, int n, int d,
                           int ntrees, int max_depth, int min_samples,
                           uint32 seed)
{
    Forest f; f.ntrees = ntrees; f.trees = palloc(sizeof(Tree) * ntrees);
    int *bag = palloc(sizeof(int) * n);
    ForestRNG rng;
    forest_rng_seed(&rng, seed);
    for (int b = 0; b < ntrees; b++) {
        for (int i = 0; i < n; i++) bag[i] = forest_rng_index(&rng, n);
        f.trees[b] = train_tree_idx(X, y, n, d, bag, max_depth, min_samples);
    }
    pfree(bag);
    return f;
}

static int predict_forest(const Forest *f, const float4 *x)
{
    int votes[64] = {0};
    for (int b = 0; b < f->ntrees; b++) { int l = predict_cls_tree(&f->trees[b], x); if (l >= 0 && l < 64) votes[l]++; }
    int best = 0, bestc = -1;
    for (int c = 0; c < 64; c++) if (votes[c] > bestc) { bestc = votes[c]; best = c; }
    return best;
}

static bytea *ser_forest(const Forest *f)
{
    int len = sizeof(int);
    for (int b = 0; b < f->ntrees; b++) len += sizeof(int) + f->trees[b].nnodes * NODE_REC;
    char *buf = palloc(len), *p = buf;
    memcpy(p, &f->ntrees, sizeof(int)); p += sizeof(int);
    for (int b = 0; b < f->ntrees; b++) write_tree(&p, &f->trees[b]);
    bytea *res = make_bytea(buf, len); pfree(buf); return res;
}
static Forest deser_forest(const bytea *b)
{
    const char *p = VARDATA(b);
    Forest f; memcpy(&f.ntrees, p, sizeof(int)); p += sizeof(int);
    f.trees = palloc(sizeof(Tree) * f.ntrees);
    for (int i = 0; i < f.ntrees; i++) read_tree(&p, &f.trees[i]);
    return f;
}

/* ====================== gradient boosting (binary log-loss) ====================== */

typedef struct GBM { int nrounds; float lr; Tree *trees; } GBM;

static double sigmoid(double z) { return 1.0 / (1.0 + exp(-z)); }

static GBM train_gbm(float4 *X, int32 *y, int n, int d, int nrounds, int depth, float lr)
{
    GBM g; g.nrounds = nrounds; g.lr = lr; g.trees = palloc(sizeof(Tree) * nrounds);
    double *pred = palloc(sizeof(double) * n);
    double *resid = palloc(sizeof(double) * n);
    int *idx = palloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) { pred[i] = 0; idx[i] = i; }
    for (int r = 0; r < nrounds; r++) {
        for (int i = 0; i < n; i++) resid[i] = (double)y[i] - sigmoid(pred[i]);
        int cap = 2 * n + 16;
        Tree t; t.nodes = palloc(sizeof(TNode) * cap); t.nnodes = 0;
        build_rtree(X, resid, n, d, idx, 0, depth, t.nodes, &t.nnodes, cap);
        for (int i = 0; i < n; i++) pred[i] += lr * (double)predict_reg_tree(&t, X + i * d);
        g.trees[r] = t;
    }
    pfree(pred); pfree(resid); pfree(idx);
    return g;
}

static int predict_gbm(const GBM *g, const float4 *x)
{
    double z = 0;
    for (int r = 0; r < g->nrounds; r++) z += g->lr * (double)predict_reg_tree(&g->trees[r], x);
    return sigmoid(z) > 0.5 ? 1 : 0;
}

static bytea *ser_gbm(const GBM *g)
{
    int len = sizeof(int) + sizeof(float);
    for (int r = 0; r < g->nrounds; r++) len += sizeof(int) + g->trees[r].nnodes * NODE_REC;
    char *buf = palloc(len), *p = buf;
    memcpy(p, &g->nrounds, sizeof(int)); p += sizeof(int);
    memcpy(p, &g->lr, sizeof(float)); p += sizeof(float);
    for (int r = 0; r < g->nrounds; r++) write_tree(&p, &g->trees[r]);
    bytea *res = make_bytea(buf, len); pfree(buf); return res;
}
static GBM deser_gbm(const bytea *b)
{
    const char *p = VARDATA(b);
    GBM g; memcpy(&g.nrounds, p, sizeof(int)); p += sizeof(int);
    memcpy(&g.lr, p, sizeof(float)); p += sizeof(float);
    g.trees = palloc(sizeof(Tree) * g.nrounds);
    for (int i = 0; i < g.nrounds; i++) read_tree(&p, &g.trees[i]);
    return g;
}

/* ====================== AdaBoost (weighted stumps) ====================== */

typedef struct AdaStump { Tree stump; float alpha; } AdaStump;
typedef struct AdaModel { int nrounds; AdaStump *s; } AdaModel;

static Tree train_weighted_stump(float4 *X, int32 *y, double *w, int n, int d)
{
    /* depth-1: exhaustive single best split by weighted error */
    Tree t; t.nnodes = 0; t.nodes = palloc(sizeof(TNode) * 4);
    int me = 0; t.nnodes = 1;
    t.nodes[0].feature = -1; t.nodes[0].threshold = 0;
    t.nodes[0].left = t.nodes[0].right = -1; t.nodes[0].value = 0;
    /* leaf default = weighted majority */
    { double w0 = 0, w1 = 0; for (int i = 0; i < n; i++) { if (y[i] == 0) w0 += w[i]; else w1 += w[i]; }
      t.nodes[0].label = (w1 > w0) ? 1 : 0; }

    int best_f = -1; float best_t = 0; double best_err = 1e18;
    SortPair *sp = palloc(sizeof(SortPair) * n);
    for (int f = 0; f < d; f++) {
        for (int i = 0; i < n; i++) { sp[i].idx = i; sp[i].val = X[i*d + f]; }
        qsort(sp, n, sizeof(SortPair), cmp_sortpair);
        double wsum = 0; for (int i = 0; i < n; i++) wsum += w[i];
        double wl0 = 0, wl1 = 0, wr0 = 0, wr1 = 0;
        for (int i = 0; i < n; i++) { if (y[i] == 0) wr0 += w[i]; else wr1 += w[i]; }
        for (int i = 0; i < n - 1; i++) {
            (y[sp[i].idx] == 0 ? (wl0 += w[sp[i].idx], wr0 -= w[sp[i].idx])
                               : (wl1 += w[sp[i].idx], wr1 -= w[sp[i].idx]));
            if (sp[i].val == sp[i+1].val) continue;
            float t = 0.5f * (sp[i].val + sp[i+1].val);
            /* left predicts majority of wl, right majority of wr */
            int pl = (wl1 > wl0) ? 1 : 0, pr = (wr1 > wr0) ? 1 : 0;
            double err = (pl == 1 ? wl0 : wl1) + (pr == 1 ? wr0 : wr1);
            if (err < best_err) { best_err = err; best_f = f; best_t = t; }
        }
    }
    pfree(sp);
    if (best_f < 0) return t; /* constant stump */

    /* build a proper 3-node stump with left/right labels */
    t.nnodes = 3;
    t.nodes[0].feature = best_f; t.nodes[0].threshold = best_t;
    t.nodes[0].left = 1; t.nodes[0].right = 2; t.nodes[0].label = 0;
    t.nodes[1].feature = -1; t.nodes[1].left = t.nodes[1].right = -1;
    t.nodes[2].feature = -1; t.nodes[2].left = t.nodes[2].right = -1;
    /* compute left/right majority labels */
    int l0 = 0, l1 = 0, r0 = 0, r1 = 0;
    for (int i = 0; i < n; i++) {
        if (X[i*d + best_f] <= best_t) (y[i] == 0 ? l0++ : l1++); else (y[i] == 0 ? r0++ : r1++);
    }
    t.nodes[1].label = (l1 >= l0) ? 1 : 0;
    t.nodes[2].label = (r1 >= r0) ? 1 : 0;
    t.nodes[1].value = t.nodes[2].value = 0;
    return t;
}

static AdaModel train_adaboost(float4 *X, int32 *y, int n, int d, int nrounds)
{
    AdaModel m; m.nrounds = nrounds; m.s = palloc(sizeof(AdaStump) * nrounds);
    double *w = palloc(sizeof(double) * n);
    for (int i = 0; i < n; i++) w[i] = 1.0 / n;
    for (int r = 0; r < nrounds; r++) {
        Tree stump = train_weighted_stump(X, y, w, n, d);
        double err = 0;
        for (int i = 0; i < n; i++)
            if (predict_cls_tree(&stump, X + i*d) != y[i]) err += w[i];
        if (err >= 0.5) err = 0.499; if (err <= 0) err = 1e-9;
        double alpha = 0.5 * log((1.0 - err) / err);
        /* update weights (signed labels: 0 -> -1, 1 -> +1) */
        double zsum = 0;
        for (int i = 0; i < n; i++) {
            int ysig = (y[i] == 0) ? -1 : 1;
            int hsig = (predict_cls_tree(&stump, X + i*d) == 0) ? -1 : 1;
            w[i] *= exp(-alpha * ysig * hsig);
            zsum += w[i];
        }
        for (int i = 0; i < n; i++) w[i] /= zsum;
        m.s[r].stump = stump; m.s[r].alpha = (float)alpha;
    }
    pfree(w);
    return m;
}

static int predict_adaboost(const AdaModel *m, const float4 *x)
{
    double s = 0;
    for (int r = 0; r < m->nrounds; r++) {
        int h = predict_cls_tree(&m->s[r].stump, x) == 0 ? -1 : 1;
        s += m->s[r].alpha * h;
    }
    return s > 0 ? 1 : 0;
}

static bytea *ser_adaboost(const AdaModel *m)
{
    int len = sizeof(int);
    for (int r = 0; r < m->nrounds; r++) len += sizeof(float) + sizeof(int) + m->s[r].stump.nnodes * NODE_REC;
    char *buf = palloc(len), *p = buf;
    memcpy(p, &m->nrounds, sizeof(int)); p += sizeof(int);
    for (int r = 0; r < m->nrounds; r++) {
        memcpy(p, &m->s[r].alpha, sizeof(float)); p += sizeof(float);
        write_tree(&p, &m->s[r].stump);
    }
    bytea *res = make_bytea(buf, len); pfree(buf); return res;
}
static AdaModel deser_adaboost(const bytea *b)
{
    const char *p = VARDATA(b);
    AdaModel m; memcpy(&m.nrounds, p, sizeof(int)); p += sizeof(int);
    m.s = palloc(sizeof(AdaStump) * m.nrounds);
    for (int i = 0; i < m.nrounds; i++) {
        memcpy(&m.s[i].alpha, p, sizeof(float)); p += sizeof(float);
        read_tree(&p, &m.s[i].stump);
    }
    return m;
}

/* ====================== linear SVM (subgradient / PEGASOS-like) ====================== */

typedef struct SVMModel { int d; float b; float *w; } SVMModel;

static SVMModel train_svm(float4 *X, int32 *y, int n, int d, float lambda, int iters)
{
    SVMModel m; m.d = d; m.w = palloc0(sizeof(float) * d); m.b = 0;
    float lr = 0.05f;
    for (int it = 0; it < iters; it++) {
        for (int i = 0; i < n; i++) {
            int ysig = (y[i] == 0) ? -1 : 1;
            float z = m.b;
            for (int j = 0; j < d; j++) z += m.w[j] * X[i*d + j];
            if (ysig * z < 1.0f) {   /* hinge violation -> subgradient step */
                for (int j = 0; j < d; j++)
                    m.w[j] = (1.0f - lr * lambda) * m.w[j] + lr * ysig * X[i*d + j];
                m.b += lr * ysig;
            } else {
                for (int j = 0; j < d; j++) m.w[j] *= (1.0f - lr * lambda);
            }
        }
    }
    return m;
}

static int predict_svm(const SVMModel *m, const float4 *x)
{
    float z = m->b;
    for (int j = 0; j < m->d; j++) z += m->w[j] * x[j];
    return z > 0 ? 1 : 0;
}

static bytea *ser_svm(const SVMModel *m)
{
    int len = sizeof(int) + sizeof(float) + sizeof(float) * m->d;
    char *buf = palloc(len), *p = buf;
    memcpy(p, &m->d, sizeof(int)); p += sizeof(int);
    memcpy(p, &m->b, sizeof(float)); p += sizeof(float);
    memcpy(p, m->w, sizeof(float) * m->d); p += sizeof(float) * m->d;
    bytea *res = make_bytea(buf, len); pfree(buf); return res;
}
static SVMModel deser_svm(const bytea *b)
{
    const char *p = VARDATA(b);
    SVMModel m; memcpy(&m.d, p, sizeof(int)); p += sizeof(int);
    memcpy(&m.b, p, sizeof(float)); p += sizeof(float);
    m.w = palloc(sizeof(float) * m.d);
    memcpy(m.w, p, sizeof(float) * m.d);
    return m;
}

/* ====================== PG wrappers ====================== */

PG_FUNCTION_INFO_V1(pgml_train_tree);
Datum pgml_train_tree(PG_FUNCTION_ARGS)
{
    PgmlStageTimer st; pgml_stage_begin(&st);
    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType *lab = PG_GETARG_ARRAYTYPE_P(1);
    int max_depth = PG_GETARG_INT32(2);
    int min_samples = PG_GETARG_INT32(3);
    int n, d; float4 *X = get_matrix(arr, &n, &d); int32 *y = get_labels(lab, &n);
    st.convert_ms = pgml_stage_lap_ms(&st);
    Tree t = train_tree(X, y, n, d, max_depth, min_samples);
    st.native_ms = pgml_stage_lap_ms(&st);
    int len = sizeof(int) + t.nnodes * NODE_REC;
    char *buf = palloc(len), *p = buf; write_tree(&p, &t);
    bytea *res = make_bytea(buf, len); pfree(buf);
    st.serialize_ms = pgml_stage_lap_ms(&st);
    pgml_stage_emit(&st);
    PG_RETURN_BYTEA_P(res);
}

PG_FUNCTION_INFO_V1(pgml_predict_tree);
Datum pgml_predict_tree(PG_FUNCTION_ARGS)
{
    bytea *b = PG_GETARG_BYTEA_P(0);
    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(1);
    int d; float4 *x = get_vector(arr, &d);
    Tree t; const char *p = VARDATA(b); read_tree(&p, &t);
    PG_RETURN_INT32(predict_cls_tree(&t, x));
}

PG_FUNCTION_INFO_V1(pgml_train_forest);
Datum pgml_train_forest(PG_FUNCTION_ARGS)
{
    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType *lab = PG_GETARG_ARRAYTYPE_P(1);
    int ntrees = PG_GETARG_INT32(2);
    int max_depth = PG_GETARG_INT32(3);
    int min_samples = PG_GETARG_INT32(4);
    int n, d; float4 *X = get_matrix(arr, &n, &d); int32 *y = get_labels(lab, &n);
    /* 5-arg form: local RNG with fixed default seed (compat signature). */
    Forest f = train_forest(X, y, n, d, ntrees, max_depth, min_samples,
                            PGML_FOREST_DEFAULT_SEED);
    PG_RETURN_BYTEA_P(ser_forest(&f));
}

PG_FUNCTION_INFO_V1(pgml_train_forest_seeded);
Datum pgml_train_forest_seeded(PG_FUNCTION_ARGS)
{
    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType *lab = PG_GETARG_ARRAYTYPE_P(1);
    int ntrees = PG_GETARG_INT32(2);
    int max_depth = PG_GETARG_INT32(3);
    int min_samples = PG_GETARG_INT32(4);
    int seed = PG_GETARG_INT32(5);
    int n, d; float4 *X = get_matrix(arr, &n, &d); int32 *y = get_labels(lab, &n);
    Forest f = train_forest(X, y, n, d, ntrees, max_depth, min_samples,
                            (uint32)seed);
    PG_RETURN_BYTEA_P(ser_forest(&f));
}

PG_FUNCTION_INFO_V1(pgml_predict_forest);
Datum pgml_predict_forest(PG_FUNCTION_ARGS)
{
    bytea *b = PG_GETARG_BYTEA_P(0);
    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(1);
    int d; float4 *x = get_vector(arr, &d);
    Forest f = deser_forest(b);
    PG_RETURN_INT32(predict_forest(&f, x));
}

PG_FUNCTION_INFO_V1(pgml_train_gbm);
Datum pgml_train_gbm(PG_FUNCTION_ARGS)
{
    PgmlStageTimer st; pgml_stage_begin(&st);
    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType *lab = PG_GETARG_ARRAYTYPE_P(1);
    int nrounds = PG_GETARG_INT32(2);
    int depth = PG_GETARG_INT32(3);
    float lr = PG_GETARG_FLOAT4(4);
    int n, d; float4 *X = get_matrix(arr, &n, &d); int32 *y = get_labels(lab, &n);
    st.convert_ms = pgml_stage_lap_ms(&st);
    GBM g = train_gbm(X, y, n, d, nrounds, depth, lr);
    st.native_ms = pgml_stage_lap_ms(&st);
    bytea *res = ser_gbm(&g);
    st.serialize_ms = pgml_stage_lap_ms(&st);
    pgml_stage_emit(&st);
    PG_RETURN_BYTEA_P(res);
}

PG_FUNCTION_INFO_V1(pgml_predict_gbm);
Datum pgml_predict_gbm(PG_FUNCTION_ARGS)
{
    bytea *b = PG_GETARG_BYTEA_P(0);
    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(1);
    int d; float4 *x = get_vector(arr, &d);
    GBM g = deser_gbm(b);
    PG_RETURN_INT32(predict_gbm(&g, x));
}

PG_FUNCTION_INFO_V1(pgml_train_adaboost);
Datum pgml_train_adaboost(PG_FUNCTION_ARGS)
{
    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType *lab = PG_GETARG_ARRAYTYPE_P(1);
    int nrounds = PG_GETARG_INT32(2);
    int n, d; float4 *X = get_matrix(arr, &n, &d); int32 *y = get_labels(lab, &n);
    AdaModel m = train_adaboost(X, y, n, d, nrounds);
    PG_RETURN_BYTEA_P(ser_adaboost(&m));
}

PG_FUNCTION_INFO_V1(pgml_predict_adaboost);
Datum pgml_predict_adaboost(PG_FUNCTION_ARGS)
{
    bytea *b = PG_GETARG_BYTEA_P(0);
    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(1);
    int d; float4 *x = get_vector(arr, &d);
    AdaModel m = deser_adaboost(b);
    PG_RETURN_INT32(predict_adaboost(&m, x));
}

PG_FUNCTION_INFO_V1(pgml_train_svm);
Datum pgml_train_svm(PG_FUNCTION_ARGS)
{
    PgmlStageTimer st; pgml_stage_begin(&st);
    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType *lab = PG_GETARG_ARRAYTYPE_P(1);
    float lambda = PG_GETARG_FLOAT4(2);
    int iters = PG_GETARG_INT32(3);
    int n, d; float4 *X = get_matrix(arr, &n, &d); int32 *y = get_labels(lab, &n);
    st.convert_ms = pgml_stage_lap_ms(&st);
    SVMModel m = train_svm(X, y, n, d, lambda, iters);
    st.native_ms = pgml_stage_lap_ms(&st);
    bytea *res = ser_svm(&m);
    st.serialize_ms = pgml_stage_lap_ms(&st);
    pgml_stage_emit(&st);
    PG_RETURN_BYTEA_P(res);
}

PG_FUNCTION_INFO_V1(pgml_predict_svm);
Datum pgml_predict_svm(PG_FUNCTION_ARGS)
{
    bytea *b = PG_GETARG_BYTEA_P(0);
    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(1);
    int d; float4 *x = get_vector(arr, &d);
    SVMModel m = deser_svm(b);
    PG_RETURN_INT32(predict_svm(&m, x));
}

/* ====================== batch inference (one C call per test set) ====================== */

static ArrayType *int_array_from(int *vals, int n)
{
    Datum *out = palloc(sizeof(Datum) * n);
    for (int i = 0; i < n; i++) out[i] = Int32GetDatum(vals[i]);
    return construct_array(out, n, INT4OID, sizeof(int32), true, 'i');
}

PG_FUNCTION_INFO_V1(pgml_predict_tree_batch);
Datum pgml_predict_tree_batch(PG_FUNCTION_ARGS)
{
    bytea *b = PG_GETARG_BYTEA_P(0);
    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(1);
    int n, d; float4 *X = get_matrix(arr, &n, &d);
    Tree t; const char *p = VARDATA(b); read_tree(&p, &t);
    int *out = palloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) out[i] = predict_cls_tree(&t, X + i * d);
    PG_RETURN_ARRAYTYPE_P(int_array_from(out, n));
}

PG_FUNCTION_INFO_V1(pgml_predict_forest_batch);
Datum pgml_predict_forest_batch(PG_FUNCTION_ARGS)
{
    bytea *b = PG_GETARG_BYTEA_P(0);
    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(1);
    int n, d; float4 *X = get_matrix(arr, &n, &d);
    Forest f = deser_forest(b);
    int *out = palloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) out[i] = predict_forest(&f, X + i * d);
    PG_RETURN_ARRAYTYPE_P(int_array_from(out, n));
}

PG_FUNCTION_INFO_V1(pgml_predict_gbm_batch);
Datum pgml_predict_gbm_batch(PG_FUNCTION_ARGS)
{
    bytea *b = PG_GETARG_BYTEA_P(0);
    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(1);
    int n, d; float4 *X = get_matrix(arr, &n, &d);
    GBM g = deser_gbm(b);
    int *out = palloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) out[i] = predict_gbm(&g, X + i * d);
    PG_RETURN_ARRAYTYPE_P(int_array_from(out, n));
}

PG_FUNCTION_INFO_V1(pgml_predict_adaboost_batch);
Datum pgml_predict_adaboost_batch(PG_FUNCTION_ARGS)
{
    bytea *b = PG_GETARG_BYTEA_P(0);
    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(1);
    int n, d; float4 *X = get_matrix(arr, &n, &d);
    AdaModel m = deser_adaboost(b);
    int *out = palloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) out[i] = predict_adaboost(&m, X + i * d);
    PG_RETURN_ARRAYTYPE_P(int_array_from(out, n));
}

PG_FUNCTION_INFO_V1(pgml_predict_svm_batch);
Datum pgml_predict_svm_batch(PG_FUNCTION_ARGS)
{
    bytea *b = PG_GETARG_BYTEA_P(0);
    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(1);
    int n, d; float4 *X = get_matrix(arr, &n, &d);
    SVMModel m = deser_svm(b);
    int *out = palloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) out[i] = predict_svm(&m, X + i * d);
    PG_RETURN_ARRAYTYPE_P(int_array_from(out, n));
}

/* ====================== histogram-optimized split finding ====================== */

#define HIST_BINS 32

static int hist_split_cls(float4 *X, int32 *y, int n, int d, int *idx,
                          double parent_g, int *bf, float *bt)
{
    double best_gain = 0; int found = 0;
    for (int f = 0; f < d; f++) {
        float mn = X[idx[0]*d+f], mx = mn;
        for (int i = 1; i < n; i++) { float v = X[idx[i]*d+f]; if (v < mn) mn = v; if (v > mx) mx = v; }
        if (mx - mn < 1e-9f) continue;
        int bin[HIST_BINS][2]; memset(bin, 0, sizeof(bin));
        for (int i = 0; i < n; i++) {
            float v = X[idx[i]*d+f];
            int b = (int)((v - mn) / (mx - mn) * HIST_BINS);
            if (b >= HIST_BINS) b = HIST_BINS - 1;
            bin[b][y[idx[i]] >= 1 ? 1 : 0]++;
        }
        int L0 = 0, L1 = 0, R0 = 0, R1 = 0;
        for (int b = 0; b < HIST_BINS; b++) { R0 += bin[b][0]; R1 += bin[b][1]; }
        for (int b = 0; b < HIST_BINS - 1; b++) {
            L0 += bin[b][0]; L1 += bin[b][1]; R0 -= bin[b][0]; R1 -= bin[b][1];
            int nl = L0 + L1, nr = R0 + R1;
            if (nl == 0 || nr == 0) continue;
            double gl = 1.0 - (double)(L0*L0 + L1*L1) / ((double)nl * nl);
            double gr = 1.0 - (double)(R0*R0 + R1*R1) / ((double)nr * nr);
            double gain = parent_g - (double)nl / n * gl - (double)nr / n * gr;
            if (gain > best_gain) { best_gain = gain; *bf = f; *bt = mn + (mx - mn) * (b + 1.0f) / HIST_BINS; found = 1; }
        }
    }
    return (found && best_gain > 1e-9);
}

static int build_tree_hist(float4 *X, int32 *y, int n, int d, int *idx,
                           int depth, int max_depth, int min_samples,
                           TNode *nodes, int *nnodes, int cap)
{
    int me = (*nnodes)++;
    if (me >= cap) elog(ERROR, "pgml: tree capacity exceeded");
    nodes[me].feature = -1; nodes[me].threshold = 0;
    nodes[me].left = nodes[me].right = -1; nodes[me].value = 0;
    nodes[me].label = majority_class(y, idx, n);
    int base[64]; double pg = cls_gini(y, idx, n, base);
    if (pg <= 1e-9 || depth >= max_depth || n < min_samples) return me;
    int bf; float bt;
    if (!hist_split_cls(X, y, n, d, idx, pg, &bf, &bt)) return me;
    int *li = palloc(sizeof(int)*n), *ri = palloc(sizeof(int)*n);
    int nl = 0, nr = 0;
    for (int i = 0; i < n; i++)
        if (X[idx[i]*d + bf] <= bt) li[nl++] = idx[i]; else ri[nr++] = idx[i];
    if (nl == 0 || nr == 0) { pfree(li); pfree(ri); return me; }
    nodes[me].feature = bf; nodes[me].threshold = bt;
    nodes[me].left  = build_tree_hist(X, y, nl, d, li, depth+1, max_depth, min_samples, nodes, nnodes, cap);
    nodes[me].right = build_tree_hist(X, y, nr, d, ri, depth+1, max_depth, min_samples, nodes, nnodes, cap);
    pfree(li); pfree(ri);
    return me;
}

static int hist_split_reg(float4 *X, double *r, int n, int d, int *idx,
                          double total_var, int *bf, float *bt)
{
    double best_gain = 0; int found = 0;
    for (int f = 0; f < d; f++) {
        float mn = X[idx[0]*d+f], mx = mn;
        for (int i = 1; i < n; i++) { float v = X[idx[i]*d+f]; if (v < mn) mn = v; if (v > mx) mx = v; }
        if (mx - mn < 1e-9f) continue;
        double sum[HIST_BINS], sq[HIST_BINS]; int cnt[HIST_BINS];
        memset(sum, 0, sizeof(sum)); memset(sq, 0, sizeof(sq)); memset(cnt, 0, sizeof(cnt));
        for (int i = 0; i < n; i++) {
            float v = X[idx[i]*d+f];
            int b = (int)((v - mn) / (mx - mn) * HIST_BINS);
            if (b >= HIST_BINS) b = HIST_BINS - 1;
            double rr = r[idx[i]]; sum[b] += rr; sq[b] += rr * rr; cnt[b]++;
        }
        double sL = 0, qL = 0, sR = 0, qR = 0; int cL = 0, cR = 0;
        for (int b = 0; b < HIST_BINS; b++) { sR += sum[b]; qR += sq[b]; cR += cnt[b]; }
        for (int b = 0; b < HIST_BINS - 1; b++) {
            sL += sum[b]; qL += sq[b]; cL += cnt[b]; sR -= sum[b]; qR -= sq[b]; cR -= cnt[b];
            if (cL == 0 || cR == 0) continue;
            double vL = qL - sL * sL / cL;
            double vR = qR - sR * sR / cR;
            double gain = total_var - vL - vR;
            if (gain > best_gain) { best_gain = gain; *bf = f; *bt = mn + (mx - mn) * (b + 1.0f) / HIST_BINS; found = 1; }
        }
    }
    return (found && best_gain > 1e-12);
}

static int build_rtree_hist(float4 *X, double *r, int n, int d, int *idx,
                            int depth, int max_depth, TNode *nodes, int *nnodes, int cap)
{
    int me = (*nnodes)++;
    nodes[me].feature = -1; nodes[me].threshold = 0;
    nodes[me].left = nodes[me].right = -1; nodes[me].label = 0;
    double mean = 0; for (int i = 0; i < n; i++) mean += r[idx[i]];
    mean /= n; nodes[me].value = (float)mean;
    if (depth >= max_depth || n < 4) return me;
    double tv = 0; for (int i = 0; i < n; i++) { double e = r[idx[i]] - mean; tv += e * e; }
    if (tv <= 1e-12) return me;
    int bf; float bt;
    if (!hist_split_reg(X, r, n, d, idx, tv, &bf, &bt)) return me;
    int *li = palloc(sizeof(int)*n), *ri = palloc(sizeof(int)*n);
    int nl = 0, nr = 0;
    for (int i = 0; i < n; i++)
        if (X[idx[i]*d + bf] <= bt) li[nl++] = idx[i]; else ri[nr++] = idx[i];
    if (nl == 0 || nr == 0) { pfree(li); pfree(ri); return me; }
    nodes[me].feature = bf; nodes[me].threshold = bt;
    nodes[me].left  = build_rtree_hist(X, r, nl, d, li, depth+1, max_depth, nodes, nnodes, cap);
    nodes[me].right = build_rtree_hist(X, r, nr, d, ri, depth+1, max_depth, nodes, nnodes, cap);
    pfree(li); pfree(ri);
    return me;
}

static Tree train_tree_hist(float4 *X, int32 *y, int n, int d, int max_depth, int min_samples)
{
    int *idx = palloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) idx[i] = i;
    int cap = 2 * n + 16;
    Tree t; t.nodes = palloc(sizeof(TNode) * cap); t.nnodes = 0;
    build_tree_hist(X, y, n, d, idx, 0, max_depth, min_samples, t.nodes, &t.nnodes, cap);
    pfree(idx);
    return t;
}

static Forest train_forest_hist(float4 *X, int32 *y, int n, int d, int ntrees,
                                int max_depth, int min_samples, uint32 seed)
{
    Forest f; f.ntrees = ntrees; f.trees = palloc(sizeof(Tree) * ntrees);
    int *bag = palloc(sizeof(int) * n);
    ForestRNG rng;
    forest_rng_seed(&rng, seed);
    for (int b = 0; b < ntrees; b++) {
        for (int i = 0; i < n; i++) bag[i] = forest_rng_index(&rng, n);
        int cap = 2 * n + 16;
        Tree t; t.nodes = palloc(sizeof(TNode) * cap); t.nnodes = 0;
        build_tree_hist(X, y, n, d, bag, 0, max_depth, min_samples, t.nodes, &t.nnodes, cap);
        f.trees[b] = t;
    }
    pfree(bag);
    return f;
}

static GBM train_gbm_hist(float4 *X, int32 *y, int n, int d, int nrounds, int depth, float lr)
{
    GBM g; g.nrounds = nrounds; g.lr = lr; g.trees = palloc(sizeof(Tree) * nrounds);
    double *pred = palloc(sizeof(double) * n);
    double *resid = palloc(sizeof(double) * n);
    int *idx = palloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) { pred[i] = 0; idx[i] = i; }
    for (int r = 0; r < nrounds; r++) {
        for (int i = 0; i < n; i++) resid[i] = (double)y[i] - sigmoid(pred[i]);
        int cap = 2 * n + 16;
        Tree t; t.nodes = palloc(sizeof(TNode) * cap); t.nnodes = 0;
        build_rtree_hist(X, resid, n, d, idx, 0, depth, t.nodes, &t.nnodes, cap);
        for (int i = 0; i < n; i++) pred[i] += lr * (double)predict_reg_tree(&t, X + i * d);
        g.trees[r] = t;
    }
    pfree(pred); pfree(resid); pfree(idx);
    return g;
}

PG_FUNCTION_INFO_V1(pgml_train_tree_fast);
Datum pgml_train_tree_fast(PG_FUNCTION_ARGS)
{
    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType *lab = PG_GETARG_ARRAYTYPE_P(1);
    int max_depth = PG_GETARG_INT32(2);
    int min_samples = PG_GETARG_INT32(3);
    int n, d; float4 *X = get_matrix(arr, &n, &d); int32 *y = get_labels(lab, &n);
    Tree t = train_tree_hist(X, y, n, d, max_depth, min_samples);
    int len = sizeof(int) + t.nnodes * NODE_REC;
    char *buf = palloc(len), *p = buf; write_tree(&p, &t);
    bytea *res = make_bytea(buf, len); pfree(buf);
    PG_RETURN_BYTEA_P(res);
}

PG_FUNCTION_INFO_V1(pgml_train_forest_fast);
Datum pgml_train_forest_fast(PG_FUNCTION_ARGS)
{
    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType *lab = PG_GETARG_ARRAYTYPE_P(1);
    int ntrees = PG_GETARG_INT32(2);
    int max_depth = PG_GETARG_INT32(3);
    int min_samples = PG_GETARG_INT32(4);
    int n, d; float4 *X = get_matrix(arr, &n, &d); int32 *y = get_labels(lab, &n);
    Forest f = train_forest_hist(X, y, n, d, ntrees, max_depth, min_samples,
                                 PGML_FOREST_DEFAULT_SEED);
    PG_RETURN_BYTEA_P(ser_forest(&f));
}

PG_FUNCTION_INFO_V1(pgml_train_forest_fast_seeded);
Datum pgml_train_forest_fast_seeded(PG_FUNCTION_ARGS)
{
    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType *lab = PG_GETARG_ARRAYTYPE_P(1);
    int ntrees = PG_GETARG_INT32(2);
    int max_depth = PG_GETARG_INT32(3);
    int min_samples = PG_GETARG_INT32(4);
    int seed = PG_GETARG_INT32(5);
    int n, d; float4 *X = get_matrix(arr, &n, &d); int32 *y = get_labels(lab, &n);
    Forest f = train_forest_hist(X, y, n, d, ntrees, max_depth, min_samples,
                                 (uint32)seed);
    PG_RETURN_BYTEA_P(ser_forest(&f));
}

PG_FUNCTION_INFO_V1(pgml_train_gbm_fast);
Datum pgml_train_gbm_fast(PG_FUNCTION_ARGS)
{
    PgmlStageTimer st; pgml_stage_begin(&st);
    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType *lab = PG_GETARG_ARRAYTYPE_P(1);
    int nrounds = PG_GETARG_INT32(2);
    int depth = PG_GETARG_INT32(3);
    float lr = PG_GETARG_FLOAT4(4);
    int n, d; float4 *X = get_matrix(arr, &n, &d); int32 *y = get_labels(lab, &n);
    st.convert_ms = pgml_stage_lap_ms(&st);
    GBM g = train_gbm_hist(X, y, n, d, nrounds, depth, lr);
    st.native_ms = pgml_stage_lap_ms(&st);
    bytea *res = ser_gbm(&g);
    st.serialize_ms = pgml_stage_lap_ms(&st);
    pgml_stage_emit(&st);
    PG_RETURN_BYTEA_P(res);
}

/* ====================== one-vs-rest SVM (multi-class) ====================== */

typedef struct SVMOvr { int K; int d; float *w; float *b; } SVMOvr;

static SVMOvr train_svm_ovr(float4 *X, int32 *y, int n, int d, int K, float lambda, int iters)
{
    SVMOvr m; m.K = K; m.d = d;
    m.w = palloc0(sizeof(float) * K * d);
    m.b = palloc0(sizeof(float) * K);
    float lr = 0.05f;
    for (int k = 0; k < K; k++) {
        float *wk = m.w + k * d;
        for (int it = 0; it < iters; it++) {
            for (int i = 0; i < n; i++) {
                int ysig = (y[i] == k) ? 1 : -1;
                float z = m.b[k];
                for (int j = 0; j < d; j++) z += wk[j] * X[i*d + j];
                if (ysig * z < 1.0f) {
                    for (int j = 0; j < d; j++) wk[j] = (1.0f - lr*lambda)*wk[j] + lr*ysig*X[i*d+j];
                    m.b[k] += lr * ysig;
                } else {
                    for (int j = 0; j < d; j++) wk[j] *= (1.0f - lr*lambda);
                }
            }
        }
    }
    return m;
}

static int predict_svm_ovr(const SVMOvr *m, const float4 *x)
{
    int best = 0; float bestz = -1e30f;
    for (int k = 0; k < m->K; k++) {
        float z = m->b[k];
        for (int j = 0; j < m->d; j++) z += m->w[k*m->d + j] * x[j];
        if (z > bestz) { bestz = z; best = k; }
    }
    return best;
}

static bytea *ser_svm_ovr(const SVMOvr *m)
{
    int len = 2*sizeof(int) + m->K*sizeof(float) + m->K*m->d*sizeof(float);
    char *buf = palloc(len), *p = buf;
    memcpy(p, &m->K, sizeof(int)); p += sizeof(int);
    memcpy(p, &m->d, sizeof(int)); p += sizeof(int);
    memcpy(p, m->b, m->K*sizeof(float)); p += m->K*sizeof(float);
    memcpy(p, m->w, m->K*m->d*sizeof(float));
    bytea *res = make_bytea(buf, len); pfree(buf); return res;
}

static SVMOvr deser_svm_ovr(const bytea *b)
{
    const char *p = VARDATA(b);
    SVMOvr m; memcpy(&m.K, p, sizeof(int)); p += sizeof(int);
    memcpy(&m.d, p, sizeof(int)); p += sizeof(int);
    m.b = palloc(sizeof(float)*m.K); memcpy(m.b, p, m.K*sizeof(float)); p += m.K*sizeof(float);
    m.w = palloc(sizeof(float)*m.K*m.d); memcpy(m.w, p, m.K*m.d*sizeof(float));
    return m;
}

PG_FUNCTION_INFO_V1(pgml_train_svm_ovr);
Datum pgml_train_svm_ovr(PG_FUNCTION_ARGS)
{
    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType *lab = PG_GETARG_ARRAYTYPE_P(1);
    int K = PG_GETARG_INT32(2);
    float lambda = PG_GETARG_FLOAT4(3);
    int iters = PG_GETARG_INT32(4);
    int n, d; float4 *X = get_matrix(arr, &n, &d); int32 *y = get_labels(lab, &n);
    SVMOvr m = train_svm_ovr(X, y, n, d, K, lambda, iters);
    PG_RETURN_BYTEA_P(ser_svm_ovr(&m));
}

PG_FUNCTION_INFO_V1(pgml_predict_svm_ovr_batch);
Datum pgml_predict_svm_ovr_batch(PG_FUNCTION_ARGS)
{
    bytea *b = PG_GETARG_BYTEA_P(0);
    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(1);
    int n, d; float4 *X = get_matrix(arr, &n, &d);
    SVMOvr m = deser_svm_ovr(b);
    int *out = palloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) out[i] = predict_svm_ovr(&m, X + i*d);
    PG_RETURN_ARRAYTYPE_P(int_array_from(out, n));
}

/* ====================== categorical-feature decision tree ====================== */

static int best_cat_split(float4 *X, int32 *y, int n, int d, int *idx, int f,
                          double parent_g, float *out_cat, double *out_gain)
{
    SortPair *sp = palloc(sizeof(SortPair) * n);
    for (int i = 0; i < n; i++) { sp[i].idx = idx[i]; sp[i].val = X[idx[i]*d + f]; }
    qsort(sp, n, sizeof(SortPair), cmp_sortpair);
    int *cat = palloc(sizeof(int) * n);
    int *c0 = palloc0(sizeof(int) * n);
    int *c1 = palloc0(sizeof(int) * n);
    int K = 0;
    for (int i = 0; i < n; i++) {
        int v = (int)sp[i].val;
        if (K == 0 || v != cat[K-1]) { cat[K++] = v; }
        if (y[sp[i].idx] == 0) c0[K-1]++; else c1[K-1]++;
    }
    double best_gain = 0; int best_cat = 0; int found = 0;
    for (int k = 0; k < K; k++) {
        int nl = c0[k] + c1[k], nr = n - nl;
        if (nl == 0 || nr == 0) continue;
        double gl = 1.0 - (double)(c0[k]*c0[k] + c1[k]*c1[k]) / ((double)nl * nl);
        int R0 = 0, R1 = 0;
        for (int kk = 0; kk < K; kk++) if (kk != k) { R0 += c0[kk]; R1 += c1[kk]; }
        double gr = 1.0 - (double)(R0*R0 + R1*R1) / ((double)nr * nr);
        double gain = parent_g - (double)nl/n*gl - (double)nr/n*gr;
        if (gain > best_gain) { best_gain = gain; best_cat = cat[k]; found = 1; }
    }
    pfree(sp); pfree(cat); pfree(c0); pfree(c1);
    if (!found) return 0;
    *out_cat = (float)best_cat; *out_gain = best_gain;
    return 1;
}

static int build_tree_cat(float4 *X, int32 *y, int n, int d, int *idx,
                          int depth, int max_depth, int min_samples, int *ftypes,
                          TNode *nodes, int *nnodes, int cap)
{
    int me = (*nnodes)++;
    if (me >= cap) elog(ERROR, "pgml: tree capacity exceeded");
    nodes[me].feature = -1; nodes[me].threshold = 0;
    nodes[me].left = nodes[me].right = -1; nodes[me].value = 0;
    nodes[me].label = majority_class(y, idx, n);
    int base[64]; double pg = cls_gini(y, idx, n, base);
    if (pg <= 1e-9 || depth >= max_depth || n < min_samples) return me;

    int best_f = -1; float best_t = 0; double best_gain = 0;
    SortPair *sp = palloc(sizeof(SortPair) * n);
    for (int f = 0; f < d; f++) {
        if (ftypes[f] == 1) {
            float cat; double gain;
            if (best_cat_split(X, y, n, d, idx, f, pg, &cat, &gain) && gain > best_gain) {
                best_gain = gain; best_f = -(f + 2); best_t = cat;
            }
        } else {
            for (int i = 0; i < n; i++) { sp[i].idx = idx[i]; sp[i].val = X[idx[i]*d + f]; }
            qsort(sp, n, sizeof(SortPair), cmp_sortpair);
            int left_c[64] = {0}, right_c[64] = {0};
            for (int i = 0; i < n; i++) { int c = y[idx[i]]; if (c >= 0 && c < 64) right_c[c]++; }
            for (int i = 0; i < n - 1; i++) {
                int c = y[sp[i].idx]; if (c >= 0 && c < 64) { left_c[c]++; right_c[c]--; }
                if (sp[i].val == sp[i+1].val) continue;
                float t = 0.5f * (sp[i].val + sp[i+1].val);
                int nl = i + 1, nr = n - nl;
                double gl = 1.0, gr = 1.0;
                for (int cc = 0; cc < 64; cc++) {
                    double pl = (double)left_c[cc] / nl;  gl -= pl * pl;
                    double pr = (double)right_c[cc] / nr; gr -= pr * pr;
                }
                double gain = pg - ((double)nl/n)*gl - ((double)nr/n)*gr;
                if (gain > best_gain) { best_gain = gain; best_f = f; best_t = t; }
            }
        }
    }
    pfree(sp);
    if (best_f == -1 || best_gain <= 1e-9) return me;

    int real_f = (best_f >= 0) ? best_f : -best_f - 2;
    int is_cat = (best_f < 0);
    int *li = palloc(sizeof(int)*n), *ri = palloc(sizeof(int)*n);
    int nl = 0, nr = 0;
    for (int i = 0; i < n; i++) {
        bool go_left = is_cat ? (X[idx[i]*d + real_f] == best_t)
                              : (X[idx[i]*d + real_f] <= best_t);
        if (go_left) li[nl++] = idx[i]; else ri[nr++] = idx[i];
    }
    if (nl == 0 || nr == 0) { pfree(li); pfree(ri); return me; }
    nodes[me].feature = best_f; nodes[me].threshold = best_t;
    nodes[me].left  = build_tree_cat(X, y, nl, d, li, depth+1, max_depth, min_samples, ftypes, nodes, nnodes, cap);
    nodes[me].right = build_tree_cat(X, y, nr, d, ri, depth+1, max_depth, min_samples, ftypes, nodes, nnodes, cap);
    pfree(li); pfree(ri);
    return me;
}

static Tree train_tree_cat(float4 *X, int32 *y, int n, int d, int *ftypes, int max_depth, int min_samples)
{
    int *idx = palloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) idx[i] = i;
    int cap = 2 * n + 16;
    Tree t; t.nodes = palloc(sizeof(TNode) * cap); t.nnodes = 0;
    build_tree_cat(X, y, n, d, idx, 0, max_depth, min_samples, ftypes, t.nodes, &t.nnodes, cap);
    pfree(idx);
    return t;
}

PG_FUNCTION_INFO_V1(pgml_train_tree_cat);
Datum pgml_train_tree_cat(PG_FUNCTION_ARGS)
{
    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType *lab = PG_GETARG_ARRAYTYPE_P(1);
    int max_depth = PG_GETARG_INT32(2);
    int min_samples = PG_GETARG_INT32(3);
    ArrayType *ft = PG_GETARG_ARRAYTYPE_P(4);
    int n, d; float4 *X = get_matrix(arr, &n, &d); int32 *y = get_labels(lab, &n);
    Datum *e; bool *nulls; int nf;
    deconstruct_array(ft, INT4OID, sizeof(int32), true, 'i', &e, &nulls, &nf);
    if (nf != d) elog(ERROR, "pgml: feature_types length must equal ncols");
    int *ftypes = palloc(sizeof(int) * d);
    for (int i = 0; i < d; i++) ftypes[i] = nulls[i] ? 0 : DatumGetInt32(e[i]);
    Tree t = train_tree_cat(X, y, n, d, ftypes, max_depth, min_samples);
    int len = sizeof(int) + t.nnodes * NODE_REC;
    char *buf = palloc(len), *p = buf; write_tree(&p, &t);
    bytea *res = make_bytea(buf, len); pfree(buf);
    PG_RETURN_BYTEA_P(res);
}
