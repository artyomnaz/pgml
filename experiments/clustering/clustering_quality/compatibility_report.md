# Clustering quality compatibility report (`clustering_quality_v1`)

Canonical detailed copy: `results/clustering_quality_v1/compatibility_report.md`.

Algorithms in pgml were **not** modified. Quality tables use `pgml_dbscan` (brute).  
`pgml_dbscan_fast` ≡ brute on all 774 suite configs (see `results/.../dbscan_fast_vs_brute/`).

## Actual mlpack 4.8.0 calls (no “when available”)

```python
mlpack.kmeans(..., algorithm="naive", kmeans_plus_plus=True,
              allow_empty_clusters=True, max_iterations=300, seed=run_id)
mlpack.dbscan(input_=Xf, epsilon=eps, min_size=min_samples, naive=True)
```

`min_size` also post-filters components with size `< min_size` to noise (mlpack `dbscan_impl.hpp`).

## Clustering.jl 0.15.8 DBSCAN

Confirmed by package source + MRE under `results/clustering_quality_v1/verification/`:
neighbors of border points are marked into `cluster_mask` even when not near any core.
