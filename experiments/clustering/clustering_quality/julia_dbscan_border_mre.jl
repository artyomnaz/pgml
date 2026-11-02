#!/usr/bin/env julia
# Minimal reproducible example for Clustering.jl 0.15.8 DBSCAN expansion.
#
# Claim under test: a point that is NOT in any core's ε-neighborhood can still
# receive a cluster label, because `_dbscan` does
#   cluster_mask[neighbors] .= true
# for every explored point — including non-core (border) points.
#
# Classic Ester/sklearn/pgml DBSCAN only assigns points that appear in some
# core's ε-neighborhood (borders of cores). Neighbors of borders that are not
# themselves near a core remain noise.
using Pkg
Pkg.activate(@__DIR__)
using Clustering

const CLUSTERING_UUID = Base.UUID("aaaa29a8-35af-508c-8bc3-b662a17a0fe5")
println("Clustering.jl version: ", Pkg.dependencies()[CLUSTERING_UUID].version)

# 2×n matrix, columns = points. Positions on a line, eps=1, min_neighbors=4.
#   A=0, B=0.3, C=0.6, D=0.9  -> cores (each has >=4 neighbors within 1)
#   E=1.9                     -> border: neighbors {D,E,F} (count 3 < 4), near core D
#   F=2.9                     -> only near E; NOT near any core (dist to D = 2)
X = Float64[
    0.0  0.3  0.6  0.9  1.9  2.9;
    0.0  0.0  0.0  0.0  0.0  0.0;
]
eps = 1.0
min_neighbors = 4
n = size(X, 2)

neigh(i) = findall(j -> sum(abs2, X[:, i] .- X[:, j]) <= eps^2, 1:n)

println("Manual neighborhoods (count includes self; core iff count >= $min_neighbors):")
for i in 1:n
    ns = neigh(i)
    println("  p$i x=$(X[1,i]) N=$ns |N|=$(length(ns)) core=$(length(ns) >= min_neighbors)")
end

cores_geom = [i for i in 1:n if length(neigh(i)) >= min_neighbors]
println("geometric cores: ", cores_geom)
@assert 6 ∉ vcat([neigh(c) for c in cores_geom]...) "F must not lie in any core neighborhood"

R = dbscan(X, eps; min_neighbors=min_neighbors, min_cluster_size=1)
println("assignments (0 = noise): ", assignments(R))
for (k, c) in enumerate(R.clusters)
    println("  cluster $k core=$(c.core_indices) boundary=$(c.boundary_indices)")
end

# Source behavior (Clustering 0.15.8 src/dbscan.jl):
#   _dbscan_region_query!(neighbors, data, point, radius)
#   cluster_mask[neighbors] .= true   # <-- applies to border points too
#   if length(neighbors) >= min_neighbors
#       core_mask[point] = true
#       push unvisited neighbors onto the explore queue
#   end
#
# When core D explores E, E is queued. When E is processed, it is not a core,
# so it does not queue F — but cluster_mask[N(E)] still marks F.

f_label = assignments(R)[6]
println("point F (index 6) assignment: ", f_label)
@assert f_label != 0 "Clustering.jl assigned F via border-neighbor expansion"
println("MRE confirmed: F is clustered despite not belonging to any core ε-neighborhood.")
