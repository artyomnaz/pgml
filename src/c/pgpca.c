/* pgpca.c — In-database PCA (dimensionality reduction): covariance matrix +
 * Jacobi eigendecomposition. Native C for PostgreSQL, no external libraries.
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
    if (ARR_NDIM(arr) != 2) elog(ERROR, "pgpca: features must be a 2-D array");
    *nrows = ARR_DIMS(arr)[0]; *ncols = ARR_DIMS(arr)[1];
    Datum *e; bool *nulls; int ne;
    deconstruct_array(arr, FLOAT4OID, sizeof(float4), true, 'i', &e, &nulls, &ne);
    if (ne != (*nrows) * (*ncols)) elog(ERROR, "pgpca: ragged feature matrix");
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

static ArrayType *float_vector_from(double *vals, int n)
{
    Datum *out = palloc(sizeof(Datum) * n);
    for (int i = 0; i < n; i++) out[i] = Float4GetDatum((float)vals[i]);
    return construct_array(out, n, FLOAT4OID, sizeof(float4), true, 'i');
}

static ArrayType *float_matrix_from(double *vals, int nrows, int ncols)
{
    int dims[2] = {nrows, ncols};
    int lbs[2] = {1, 1};
    Datum *out = palloc(sizeof(Datum) * nrows * ncols);
    for (int i = 0; i < nrows * ncols; i++) out[i] = Float4GetDatum((float)vals[i]);
    return construct_md_array(out, NULL, 2, dims, lbs, FLOAT4OID, sizeof(float4), true, 'i');
}

/* ====================== Jacobi eigendecomposition (symmetric) ====================== */

static void jacobi(double *A, double *V, int n, int max_sweeps)
{
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) V[i*n + j] = (i == j) ? 1.0 : 0.0;
    for (int sweep = 0; sweep < max_sweeps; sweep++) {
        double off = 0;
        for (int p = 0; p < n; p++) for (int q = p + 1; q < n; q++) off += A[p*n+q] * A[p*n+q];
        if (off < 1e-20) break;
        for (int p = 0; p < n; p++) {
            for (int q = p + 1; q < n; q++) {
                double apq = A[p*n+q];
                if (fabs(apq) < 1e-15) continue;
                double app = A[p*n+p], aqq = A[q*n+q];
                double tau = (aqq - app) / (2.0 * apq);
                double t = (tau >= 0 ? 1.0 : -1.0) / (fabs(tau) + sqrt(1.0 + tau*tau));
                double c = 1.0 / sqrt(1.0 + t*t), s = t * c;
                A[p*n+p] = app - t * apq;
                A[q*n+q] = aqq + t * apq;
                A[p*n+q] = A[q*n+p] = 0.0;
                for (int k = 0; k < n; k++) {
                    if (k == p || k == q) continue;
                    double akp = A[k*n+p], akq = A[k*n+q];
                    A[k*n+p] = A[p*n+k] = c*akp - s*akq;
                    A[k*n+q] = A[q*n+k] = s*akp + c*akq;
                }
                for (int k = 0; k < n; k++) {
                    double vkp = V[k*n+p], vkq = V[k*n+q];
                    V[k*n+p] = c*vkp - s*vkq;
                    V[k*n+q] = s*vkp + c*vkq;
                }
            }
        }
    }
}

/* ====================== PCA ====================== */

typedef struct PCA { int d; int k; float *mean; float *evals; float *evecs; } PCA;

static PCA pca_fit(float4 *X, int n, int d, int k)
{
    /* center */
    float *mean = palloc0(sizeof(float) * d);
    for (int i = 0; i < n; i++) for (int j = 0; j < d; j++) mean[j] += X[i*d + j];
    for (int j = 0; j < d; j++) mean[j] /= n;

    /* covariance C = Xc^T Xc / (n-1) */
    double *C = palloc0(sizeof(double) * d * d);
    for (int i = 0; i < n; i++) {
        for (int p = 0; p < d; p++) {
            double xp = X[i*d + p] - mean[p];
            for (int q = p; q < d; q++) {
                double xq = X[i*d + q] - mean[q];
                C[p*d + q] += xp * xq;
            }
        }
    }
    for (int p = 0; p < d; p++) for (int q = p; q < d; q++) {
        C[p*d + q] /= (n - 1);
        C[q*d + p] = C[p*d + q];
    }

    /* eigendecompose */
    double *V = palloc(sizeof(double) * d * d);
    jacobi(C, V, d, 100);
    /* C diagonal = eigenvalues (in V column order) */

    /* sort eigenvalues descending, permute eigenvectors */
    double *evals = palloc(sizeof(double) * d);
    int *order = palloc(sizeof(int) * d);
    for (int j = 0; j < d; j++) { evals[j] = C[j*d + j]; order[j] = j; }
    for (int a = 0; a < d; a++)
        for (int b = a + 1; b < d; b++)
            if (evals[order[b]] > evals[order[a]]) { int t = order[a]; order[a] = order[b]; order[b] = t; }

    PCA m; m.d = d; m.k = k;
    m.mean = palloc(sizeof(float) * d);
    m.evals = palloc(sizeof(float) * d);
    m.evecs = palloc(sizeof(float) * d * k);
    for (int j = 0; j < d; j++) {
        m.mean[j] = mean[j];
        m.evals[j] = (float)evals[order[j]];
    }
    for (int j = 0; j < d; j++)
        for (int c = 0; c < k; c++)
            m.evecs[j*k + c] = (float)V[j*d + order[c]];

    pfree(mean); pfree(C); pfree(V); pfree(evals); pfree(order);
    return m;
}

static void pca_transform(const PCA *m, const float4 *X, int n, double *Z)
{
    for (int i = 0; i < n; i++)
        for (int c = 0; c < m->k; c++) {
            double z = 0;
            for (int j = 0; j < m->d; j++) z += (X[i*m->d + j] - m->mean[j]) * m->evecs[j*m->k + c];
            Z[i*m->k + c] = z;
        }
}

static bytea *ser_pca(const PCA *m)
{
    int len = 2 * sizeof(int) + m->d * sizeof(float) * (2 + m->k);
    char *buf = palloc(len), *p = buf;
    memcpy(p, &m->d, sizeof(int)); p += sizeof(int);
    memcpy(p, &m->k, sizeof(int)); p += sizeof(int);
    memcpy(p, m->mean, m->d * sizeof(float)); p += m->d * sizeof(float);
    memcpy(p, m->evals, m->d * sizeof(float)); p += m->d * sizeof(float);
    memcpy(p, m->evecs, m->d * m->k * sizeof(float));
    return make_bytea(buf, len);
}

static PCA deser_pca(const bytea *b)
{
    const char *p = VARDATA(b);
    PCA m; memcpy(&m.d, p, sizeof(int)); p += sizeof(int);
    memcpy(&m.k, p, sizeof(int)); p += sizeof(int);
    m.mean = palloc(sizeof(float) * m.d); memcpy(m.mean, p, m.d * sizeof(float)); p += m.d * sizeof(float);
    m.evals = palloc(sizeof(float) * m.d); memcpy(m.evals, p, m.d * sizeof(float)); p += m.d * sizeof(float);
    m.evecs = palloc(sizeof(float) * m.d * m.k); memcpy(m.evecs, p, m.d * m.k * sizeof(float));
    return m;
}

/* ====================== PG wrappers ====================== */

PG_FUNCTION_INFO_V1(pgpca_fit);
Datum pgpca_fit(PG_FUNCTION_ARGS)
{
    PgmlStageTimer st; pgml_stage_begin(&st);
    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(0);
    int k = PG_GETARG_INT32(1);
    int n, d; float4 *X = get_matrix(arr, &n, &d);
    st.convert_ms = pgml_stage_lap_ms(&st);
    if (k > d) k = d;
    PCA m = pca_fit(X, n, d, k);
    st.native_ms = pgml_stage_lap_ms(&st);
    bytea *res = ser_pca(&m);
    st.serialize_ms = pgml_stage_lap_ms(&st);
    pgml_stage_emit(&st);
    PG_RETURN_BYTEA_P(res);
}

PG_FUNCTION_INFO_V1(pgpca_transform);
Datum pgpca_transform(PG_FUNCTION_ARGS)
{
    bytea *b = PG_GETARG_BYTEA_P(0);
    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(1);
    int n, d; float4 *X = get_matrix(arr, &n, &d);
    PCA m = deser_pca(b);
    double *Z = palloc(sizeof(double) * n * m.k);
    pca_transform(&m, X, n, Z);
    PG_RETURN_ARRAYTYPE_P(float_matrix_from(Z, n, m.k));
}

PG_FUNCTION_INFO_V1(pgpca_eigenvalues);
Datum pgpca_eigenvalues(PG_FUNCTION_ARGS)
{
    bytea *b = PG_GETARG_BYTEA_P(0);
    PCA m = deser_pca(b);
    double *e = palloc(sizeof(double) * m.d);
    for (int j = 0; j < m.d; j++) e[j] = m.evals[j];
    PG_RETURN_ARRAYTYPE_P(float_vector_from(e, m.d));
}

/* ====================== multithreaded PCA (parallel covariance) ====================== */

typedef struct { const float *X; const float *mean; int n, d, start, end; double *C; } PCAJob;

static void *pca_worker(void *arg)
{
    PCAJob *j = arg;
    for (int i = j->start; i < j->end; i++)
        for (int p = 0; p < j->d; p++) {
            double xp = j->X[i*j->d + p] - j->mean[p];
            for (int q = p; q < j->d; q++) {
                double xq = j->X[i*j->d + q] - j->mean[q];
                j->C[p*j->d + q] += xp * xq;
            }
        }
    return NULL;
}

static double *pca_cov_parallel(const float4 *X, const float *mean, int n, int d, int n_threads)
{
    if (n_threads < 1) n_threads = 1;
    if (n_threads > n) n_threads = n;
    pthread_t *th = malloc(sizeof(pthread_t) * n_threads);
    PCAJob *jobs = malloc(sizeof(PCAJob) * n_threads);
    double **Cs = malloc(sizeof(double *) * n_threads);
    for (int t = 0; t < n_threads; t++) Cs[t] = calloc(d * d, sizeof(double));
    int chunk = (n + n_threads - 1) / n_threads;
    for (int t = 0; t < n_threads; t++) {
        jobs[t].X = X; jobs[t].mean = mean; jobs[t].n = n; jobs[t].d = d;
        jobs[t].start = t * chunk;
        jobs[t].end = ((t + 1) * chunk < n) ? (t + 1) * chunk : n;
        jobs[t].C = Cs[t];
        pthread_create(&th[t], NULL, pca_worker, &jobs[t]);
    }
    for (int t = 0; t < n_threads; t++) pthread_join(th[t], NULL);
    double *C = palloc0(sizeof(double) * d * d);
    for (int t = 0; t < n_threads; t++)
        for (int i = 0; i < d * d; i++) C[i] += Cs[t][i];
    for (int t = 0; t < n_threads; t++) free(Cs[t]);
    free(Cs); free(jobs); free(th);
    return C;
}

static PCA pca_fit_parallel(float4 *X, int n, int d, int k, int n_threads)
{
    float *mean = palloc0(sizeof(float) * d);
    for (int i = 0; i < n; i++) for (int j = 0; j < d; j++) mean[j] += X[i*d + j];
    for (int j = 0; j < d; j++) mean[j] /= n;
    double *C = pca_cov_parallel(X, mean, n, d, n_threads);
    for (int p = 0; p < d; p++) for (int q = p; q < d; q++) {
        C[p*d + q] /= (n - 1); C[q*d + p] = C[p*d + q];
    }
    double *V = palloc(sizeof(double) * d * d);
    jacobi(C, V, d, 100);
    double *evals = palloc(sizeof(double) * d);
    int *order = palloc(sizeof(int) * d);
    for (int j = 0; j < d; j++) { evals[j] = C[j*d + j]; order[j] = j; }
    for (int a = 0; a < d; a++) for (int b = a + 1; b < d; b++)
        if (evals[order[b]] > evals[order[a]]) { int t = order[a]; order[a] = order[b]; order[b] = t; }
    PCA m; m.d = d; m.k = k;
    m.mean = palloc(sizeof(float) * d); m.evals = palloc(sizeof(float) * d); m.evecs = palloc(sizeof(float) * d * k);
    for (int j = 0; j < d; j++) { m.mean[j] = mean[j]; m.evals[j] = (float)evals[order[j]]; }
    for (int j = 0; j < d; j++) for (int c = 0; c < k; c++) m.evecs[j*k + c] = (float)V[j*d + order[c]];
    pfree(mean); pfree(C); pfree(V); pfree(evals); pfree(order);
    return m;
}

PG_FUNCTION_INFO_V1(pgpca_fit_parallel);
Datum pgpca_fit_parallel(PG_FUNCTION_ARGS)
{
    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(0);
    int k = PG_GETARG_INT32(1);
    int n_threads = PG_GETARG_INT32(2);
    int n, d; float4 *X = get_matrix(arr, &n, &d);
    if (k > d) k = d;
    PCA m = pca_fit_parallel(X, n, d, k, n_threads);
    PG_RETURN_BYTEA_P(ser_pca(&m));
}
