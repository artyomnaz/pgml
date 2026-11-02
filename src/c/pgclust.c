/* pgclust.c — In-database clustering: k-means (k-means++ / Lloyd's), DBSCAN
 * (brute-force + grid-index), agglomerative hierarchical (min/max linkage).
 * Native C for PostgreSQL, no external libraries. Models serialize to bytea.
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
    if (ARR_NDIM(arr) != 2)
        elog(ERROR, "pgclust: features must be a 2-D array");
    *nrows = ARR_DIMS(arr)[0];
    *ncols = ARR_DIMS(arr)[1];
    Datum *e; bool *nulls; int ne;
    deconstruct_array(arr, FLOAT4OID, sizeof(float4), true, 'i', &e, &nulls, &ne);
    if (ne != (*nrows) * (*ncols)) elog(ERROR, "pgclust: ragged feature matrix");
    float4 *X = palloc(sizeof(float4) * ne);
    for (int i = 0; i < ne; i++) X[i] = nulls[i] ? 0.0f : DatumGetFloat4(e[i]);
    return X;
}

static bytea *make_bytea(const void *data, int len)
{
    bytea *b = palloc(VARHDRSZ + len);
    SET_VARSIZE(b, VARHDRSZ + len);
    if (len > 0) memcpy(VARDATA(b), data, len);
    return b;
}

static ArrayType *int_array_from(int *vals, int n)
{
    Datum *out = palloc(sizeof(Datum) * n);
    for (int i = 0; i < n; i++) out[i] = Int32GetDatum(vals[i]);
    return construct_array(out, n, INT4OID, sizeof(int32), true, 'i');
}

static inline double sqdist(const float *a, const float *b, int d)
{
    double s = 0;
    for (int j = 0; j < d; j++) { double e = (double)a[j] - b[j]; s += e * e; }
    return s;
}

/* deterministic LCG */
static uint32 rng_state;
static void rng_seed(uint32 s) { rng_state = s ? s : 1; }
static uint32 rng_next(void) { rng_state = rng_state * 1664525u + 1013904223u; return rng_state; }
static double rng_uniform(void) { return rng_next() / 4294967295.0; }

/* ====================== k-means (k-means++ / Lloyd's) ====================== */

typedef struct KMeans { int k; int d; float *centroids; } KMeans;

static KMeans train_kmeans(float4 *X, int n, int d, int k, int iters, uint32 seed, bool use_plusplus)
{
    KMeans m; m.k = k; m.d = d; m.centroids = palloc0(sizeof(float) * k * d);
    int *assign = palloc(sizeof(int) * n);
    memset(assign, 0, sizeof(int) * n);
    rng_seed(seed);

    if (use_plusplus) {
        int first = (int)(rng_uniform() * n);
        memcpy(m.centroids, X + first * d, sizeof(float) * d);
        for (int c = 1; c < k; c++) {
            double *D2 = palloc(sizeof(double) * n);
            double total = 0;
            for (int i = 0; i < n; i++) {
                double best = 1e300;
                for (int cc = 0; cc < c; cc++) {
                    double dd = sqdist(X + i*d, m.centroids + cc*d, d);
                    if (dd < best) best = dd;
                }
                D2[i] = best; total += best;
            }
            double r = rng_uniform() * total, cum = 0; int sel = n - 1;
            for (int i = 0; i < n; i++) { cum += D2[i]; if (cum >= r) { sel = i; break; } }
            memcpy(m.centroids + c*d, X + sel*d, sizeof(float) * d);
            pfree(D2);
        }
    } else {
        for (int c = 0; c < k; c++) {
            int sel = (int)((seed + c * 2654435761u) % n);
            memcpy(m.centroids + c*d, X + sel*d, sizeof(float) * d);
        }
    }

    for (int it = 0; it < iters; it++) {
        int changed = 0;
        for (int i = 0; i < n; i++) {
            int bc = 0; double bd = 1e300;
            for (int c = 0; c < k; c++) {
                double dd = sqdist(X + i*d, m.centroids + c*d, d);
                if (dd < bd) { bd = dd; bc = c; }
            }
            if (assign[i] != bc) { assign[i] = bc; changed++; }
        }
        double *sum = palloc0(sizeof(double) * k * d);
        int *cnt = palloc0(sizeof(int) * k);
        for (int i = 0; i < n; i++) {
            cnt[assign[i]]++;
            for (int j = 0; j < d; j++) sum[assign[i]*d + j] += X[i*d + j];
        }
        for (int c = 0; c < k; c++)
            if (cnt[c] > 0) for (int j = 0; j < d; j++) m.centroids[c*d + j] = (float)(sum[c*d + j] / cnt[c]);
        pfree(sum); pfree(cnt);
        if (changed == 0) break;
    }
    pfree(assign);
    return m;
}

static int predict_kmeans(const KMeans *m, const float *x)
{
    int bc = 0; double bd = 1e300;
    for (int c = 0; c < m->k; c++) {
        double dd = sqdist(x, m->centroids + c * m->d, m->d);
        if (dd < bd) { bd = dd; bc = c; }
    }
    return bc;
}

static bytea *ser_kmeans(const KMeans *m)
{
    int len = 2 * sizeof(int) + m->k * m->d * sizeof(float);
    char *buf = palloc(len), *p = buf;
    memcpy(p, &m->k, sizeof(int)); p += sizeof(int);
    memcpy(p, &m->d, sizeof(int)); p += sizeof(int);
    memcpy(p, m->centroids, m->k * m->d * sizeof(float));
    return make_bytea(buf, len);
}

static KMeans deser_kmeans(const bytea *b)
{
    const char *p = VARDATA(b);
    KMeans m; memcpy(&m.k, p, sizeof(int)); p += sizeof(int);
    memcpy(&m.d, p, sizeof(int)); p += sizeof(int);
    m.centroids = palloc(sizeof(float) * m.k * m.d);
    memcpy(m.centroids, p, m.k * m.d * sizeof(float));
    return m;
}

/* ====================== DBSCAN (brute force) ====================== */

static int32 *dbscan_brute(float4 *X, int n, int d, float eps, int min_pts)
{
    float e2 = eps * eps;
    bool *is_core = palloc(sizeof(bool) * n);
    for (int i = 0; i < n; i++) {
        int nn = 0;
        for (int j = 0; j < n; j++) if (sqdist(X + i*d, X + j*d, d) <= e2) nn++;
        is_core[i] = (nn >= min_pts);
    }
    int32 *label = palloc(sizeof(int32) * n);
    for (int i = 0; i < n; i++) label[i] = -1;
    int *stack = palloc(sizeof(int) * n);
    int cluster = 0;
    for (int i = 0; i < n; i++) {
        if (label[i] != -1 || !is_core[i]) continue;
        label[i] = cluster;
        int top = 0;
        for (int j = 0; j < n; j++)
            if (label[j] == -1 && sqdist(X + i*d, X + j*d, d) <= e2) { label[j] = cluster; stack[top++] = j; }
        while (top > 0) {
            int j = stack[--top];
            if (is_core[j])
                for (int jj = 0; jj < n; jj++)
                    if (label[jj] == -1 && sqdist(X + j*d, X + jj*d, d) <= e2) { label[jj] = cluster; stack[top++] = jj; }
        }
        cluster++;
    }
    pfree(is_core); pfree(stack);
    return label;
}

/* ====================== DBSCAN (2-D grid index, fast for low-d) ====================== */

typedef struct { uint64 key; int idx; } CellEntry;
static int cmp_cell(const void *a, const void *b)
{
    const CellEntry *x = a, *y = b;
    return (x->key > y->key) - (x->key < y->key);
}

static int grid_cell_lo(const CellEntry *ce, int n, uint64 key)
{
    int lo = 0, hi = n;
    while (lo < hi) { int m = (lo + hi) / 2; if (ce[m].key < key) lo = m + 1; else hi = m; }
    return lo;
}

static int32 *dbscan_grid2d(float4 *X, int n, float eps, int min_pts)
{
    float e2 = eps * eps;
    float mnx = X[0], mxx = X[0], mny = X[1], mxy = X[1];
    for (int i = 0; i < n; i++) {
        if (X[i*2] < mnx) mnx = X[i*2]; if (X[i*2] > mxx) mxx = X[i*2];
        if (X[i*2+1] < mny) mny = X[i*2+1]; if (X[i*2+1] > mxy) mxy = X[i*2+1];
    }
    int *cix = palloc(sizeof(int) * n), *ciy = palloc(sizeof(int) * n);
    CellEntry *ce = palloc(sizeof(CellEntry) * n);
    for (int i = 0; i < n; i++) {
        cix[i] = (int)floor((X[i*2] - mnx) / eps);
        ciy[i] = (int)floor((X[i*2+1] - mny) / eps);
        ce[i].key = ((uint64)(uint32)cix[i] << 32) | (uint32)ciy[i];
        ce[i].idx = i;
    }
    qsort(ce, n, sizeof(CellEntry), cmp_cell);

    bool *is_core = palloc(sizeof(bool) * n);
    for (int i = 0; i < n; i++) {
        int nn = 0;
        for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++) {
                uint64 key = ((uint64)(uint32)(cix[i]+dx) << 32) | (uint32)(ciy[i]+dy);
                int lo = grid_cell_lo(ce, n, key);
                for (int t = lo; t < n && ce[t].key == key; t++) {
                    int j = ce[t].idx;
                    if (sqdist(X + i*2, X + j*2, 2) <= e2) nn++;
                }
            }
        is_core[i] = (nn >= min_pts);
    }
    int32 *label = palloc(sizeof(int32) * n);
    for (int i = 0; i < n; i++) label[i] = -1;
    int *stack = palloc(sizeof(int) * n);
    int cluster = 0;
    for (int i = 0; i < n; i++) {
        if (label[i] != -1 || !is_core[i]) continue;
        label[i] = cluster;
        int top = 0;
        for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++) {
                uint64 key = ((uint64)(uint32)(cix[i]+dx) << 32) | (uint32)(ciy[i]+dy);
                int lo = grid_cell_lo(ce, n, key);
                for (int t = lo; t < n && ce[t].key == key; t++) {
                    int j = ce[t].idx;
                    if (j == i) continue;
                    if (label[j] == -1 && sqdist(X + i*2, X + j*2, 2) <= e2) { label[j] = cluster; stack[top++] = j; }
                }
            }
        while (top > 0) {
            int j = stack[--top];
            if (is_core[j])
                for (int dx = -1; dx <= 1; dx++)
                    for (int dy = -1; dy <= 1; dy++) {
                        uint64 key = ((uint64)(uint32)(cix[j]+dx) << 32) | (uint32)(ciy[j]+dy);
                        int lo = grid_cell_lo(ce, n, key);
                        for (int t = lo; t < n && ce[t].key == key; t++) {
                            int jj = ce[t].idx;
                            if (label[jj] == -1 && sqdist(X + j*2, X + jj*2, 2) <= e2) { label[jj] = cluster; stack[top++] = jj; }
                        }
                    }
        }
        cluster++;
    }
    pfree(cix); pfree(ciy); pfree(ce); pfree(is_core); pfree(stack);
    return label;
}

/* ====================== agglomerative hierarchical ====================== */

static int32 *hierarchical(float4 *X, int n, int d, int k, int linkage)
{
    /* distance matrix (upper triangular, n x n) */
    double *D = palloc(sizeof(double) * n * n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            D[i*n + j] = (i == j) ? 0.0 : sqdist(X + i*d, X + j*d, d);

    int *cl = palloc(sizeof(int) * n);   /* cluster id per point */
    int *cnt = palloc(sizeof(int) * n);  /* size per cluster */
    bool *active = palloc(sizeof(bool) * n);
    for (int i = 0; i < n; i++) { cl[i] = i; cnt[i] = 1; active[i] = true; }
    int nclust = n;

    while (nclust > k) {
        /* find closest pair of active clusters */
        int bi = -1, bj = -1; double bd = 1e300;
        for (int i = 0; i < n; i++) {
            if (!active[i]) continue;
            for (int j = i + 1; j < n; j++) {
                if (!active[j]) continue;
                if (D[i*n + j] < bd) { bd = D[i*n + j]; bi = i; bj = j; }
            }
        }
        if (bi < 0) break;
        /* merge bj into bi (cluster bi absorbs bj) */
        for (int m = 0; m < n; m++) {
            if (!active[m] || m == bi || m == bj) continue;
            double dij = D[bi*n + m], djj = D[bj*n + m];
            double dm = (linkage == 0) ? (dij < djj ? dij : djj) : (dij > djj ? dij : djj);
            D[bi*n + m] = D[m*n + bi] = dm;
        }
        /* relabel points of bj to bi */
        for (int p = 0; p < n; p++) if (cl[p] == bj) cl[p] = bi;
        cnt[bi] += cnt[bj];
        active[bj] = false;
        nclust--;
    }

    /* compress cluster ids to 0..k-1 */
    int next = 0;
    int *map = palloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) map[i] = -1;
    for (int p = 0; p < n; p++) {
        int cid = cl[p];
        if (map[cid] < 0) map[cid] = next++;
        cl[p] = map[cid];
    }
    pfree(D); pfree(cnt); pfree(active); pfree(map);
    return cl;
}

/* ====================== PG wrappers ====================== */

PG_FUNCTION_INFO_V1(pgclust_kmeans);
Datum pgclust_kmeans(PG_FUNCTION_ARGS)
{
    PgmlStageTimer st; pgml_stage_begin(&st);
    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(0);
    int k = PG_GETARG_INT32(1);
    int iters = PG_GETARG_INT32(2);
    int seed = PG_GETARG_INT32(3);
    int n, d; float4 *X = get_matrix(arr, &n, &d);
    st.convert_ms = pgml_stage_lap_ms(&st);
    KMeans m = train_kmeans(X, n, d, k, iters, (uint32)seed, true);
    st.native_ms = pgml_stage_lap_ms(&st);
    bytea *res = ser_kmeans(&m);
    st.serialize_ms = pgml_stage_lap_ms(&st);
    pgml_stage_emit(&st);
    PG_RETURN_BYTEA_P(res);
}

PG_FUNCTION_INFO_V1(pgclust_kmeans_random);
Datum pgclust_kmeans_random(PG_FUNCTION_ARGS)
{
    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(0);
    int k = PG_GETARG_INT32(1);
    int iters = PG_GETARG_INT32(2);
    int seed = PG_GETARG_INT32(3);
    int n, d; float4 *X = get_matrix(arr, &n, &d);
    KMeans m = train_kmeans(X, n, d, k, iters, (uint32)seed, false);
    PG_RETURN_BYTEA_P(ser_kmeans(&m));
}

PG_FUNCTION_INFO_V1(pgclust_kmeans_predict_batch);
Datum pgclust_kmeans_predict_batch(PG_FUNCTION_ARGS)
{
    bytea *b = PG_GETARG_BYTEA_P(0);
    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(1);
    int n, d; float4 *X = get_matrix(arr, &n, &d);
    KMeans m = deser_kmeans(b);
    int *out = palloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) out[i] = predict_kmeans(&m, X + i*d);
    PG_RETURN_ARRAYTYPE_P(int_array_from(out, n));
}

PG_FUNCTION_INFO_V1(pgclust_dbscan);
Datum pgclust_dbscan(PG_FUNCTION_ARGS)
{
    PgmlStageTimer st; pgml_stage_begin(&st);
    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(0);
    float eps = PG_GETARG_FLOAT4(1);
    int min_pts = PG_GETARG_INT32(2);
    int n, d; float4 *X = get_matrix(arr, &n, &d);
    st.convert_ms = pgml_stage_lap_ms(&st);
    int32 *label = dbscan_brute(X, n, d, eps, min_pts);
    st.native_ms = pgml_stage_lap_ms(&st);
    int *out = palloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) out[i] = label[i];
    ArrayType *res = int_array_from(out, n);
    st.serialize_ms = pgml_stage_lap_ms(&st);
    pgml_stage_emit(&st);
    PG_RETURN_ARRAYTYPE_P(res);
}

PG_FUNCTION_INFO_V1(pgclust_dbscan_fast);
Datum pgclust_dbscan_fast(PG_FUNCTION_ARGS)
{
    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(0);
    float eps = PG_GETARG_FLOAT4(1);
    int min_pts = PG_GETARG_INT32(2);
    int n, d; float4 *X = get_matrix(arr, &n, &d);
    int32 *label = (d == 2) ? dbscan_grid2d(X, n, eps, min_pts) : dbscan_brute(X, n, d, eps, min_pts);
    int *out = palloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) out[i] = label[i];
    PG_RETURN_ARRAYTYPE_P(int_array_from(out, n));
}

PG_FUNCTION_INFO_V1(pgclust_hierarchical);
Datum pgclust_hierarchical(PG_FUNCTION_ARGS)
{
    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(0);
    int k = PG_GETARG_INT32(1);
    int linkage = PG_GETARG_INT32(2);
    int n, d; float4 *X = get_matrix(arr, &n, &d);
    int32 *label = hierarchical(X, n, d, k, linkage);
    int *out = palloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) out[i] = label[i];
    PG_RETURN_ARRAYTYPE_P(int_array_from(out, n));
}

/* ====================== multithreaded k-means ====================== */

typedef struct { const float *X; const float *C; int n, d, k, start, end; int *assign; } KMJob;

static void *km_worker(void *arg)
{
    KMJob *j = arg;
    for (int i = j->start; i < j->end; i++) {
        int bc = 0; double bd = 1e300;
        for (int c = 0; c < j->k; c++) {
            double dd = sqdist(j->X + i*j->d, j->C + c*j->d, j->d);
            if (dd < bd) { bd = dd; bc = c; }
        }
        j->assign[i] = bc;
    }
    return NULL;
}

static KMeans train_kmeans_parallel(float4 *X, int n, int d, int k, int iters, uint32 seed, int n_threads)
{
    KMeans m; m.k = k; m.d = d; m.centroids = palloc0(sizeof(float) * k * d);
    int *assign = malloc(sizeof(int) * n);   /* malloc: written by worker threads */
    memset(assign, 0, sizeof(int) * n);
    rng_seed(seed);
    /* k-means++ init (single-threaded) */
    int first = (int)(rng_uniform() * n);
    memcpy(m.centroids, X + first * d, sizeof(float) * d);
    for (int c = 1; c < k; c++) {
        double *D2 = palloc(sizeof(double) * n);
        double total = 0;
        for (int i = 0; i < n; i++) {
            double best = 1e300;
            for (int cc = 0; cc < c; cc++) {
                double dd = sqdist(X + i*d, m.centroids + cc*d, d);
                if (dd < best) best = dd;
            }
            D2[i] = best; total += best;
        }
        double r = rng_uniform() * total, cum = 0; int sel = n - 1;
        for (int i = 0; i < n; i++) { cum += D2[i]; if (cum >= r) { sel = i; break; } }
        memcpy(m.centroids + c*d, X + sel*d, sizeof(float) * d);
        pfree(D2);
    }
    if (n_threads < 1) n_threads = 1;
    if (n_threads > n) n_threads = n;
    pthread_t *th = malloc(sizeof(pthread_t) * n_threads);
    KMJob *jobs = malloc(sizeof(KMJob) * n_threads);
    for (int it = 0; it < iters; it++) {
        int chunk = (n + n_threads - 1) / n_threads;
        for (int t = 0; t < n_threads; t++) {
            jobs[t].X = X; jobs[t].C = m.centroids;
            jobs[t].n = n; jobs[t].d = d; jobs[t].k = k;
            jobs[t].start = t * chunk;
            jobs[t].end = ((t + 1) * chunk < n) ? (t + 1) * chunk : n;
            jobs[t].assign = assign;
            pthread_create(&th[t], NULL, km_worker, &jobs[t]);
        }
        for (int t = 0; t < n_threads; t++) pthread_join(th[t], NULL);
        double *sum = palloc0(sizeof(double) * k * d);
        int *cnt = palloc0(sizeof(int) * k);
        for (int i = 0; i < n; i++) {
            cnt[assign[i]]++;
            for (int j = 0; j < d; j++) sum[assign[i]*d + j] += X[i*d + j];
        }
        for (int c = 0; c < k; c++)
            if (cnt[c] > 0) for (int j = 0; j < d; j++) m.centroids[c*d + j] = (float)(sum[c*d + j] / cnt[c]);
        pfree(sum); pfree(cnt);
    }
    free(assign); free(th); free(jobs);
    return m;
}

PG_FUNCTION_INFO_V1(pgclust_kmeans_parallel);
Datum pgclust_kmeans_parallel(PG_FUNCTION_ARGS)
{
    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(0);
    int k = PG_GETARG_INT32(1);
    int iters = PG_GETARG_INT32(2);
    int seed = PG_GETARG_INT32(3);
    int n_threads = PG_GETARG_INT32(4);
    int n, d; float4 *X = get_matrix(arr, &n, &d);
    KMeans m = train_kmeans_parallel(X, n, d, k, iters, (uint32)seed, n_threads);
    PG_RETURN_BYTEA_P(ser_kmeans(&m));
}
