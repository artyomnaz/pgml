#!/usr/bin/env julia
# Read jobs JSON array; each job: algo, X_csv, out_json, k, seed, eps, min_samples
using Pkg
Pkg.activate(@__DIR__)
using Clustering
using Distances
using JSON
using Random
using DelimitedFiles

function run_job(job)
    algo = job["algo"]
    x_path = job["X"]
    out_path = job["out"]
    k = Int(get(job, "k", 0))
    seed = Int(get(job, "seed", 0))
    eps = Float64(get(job, "eps", 0.3))
    min_samples = Int(get(job, "min_samples", 5))

    Xrow = readdlm(x_path, ',', Float64)
    n, d = size(Xrow)
    X = Matrix(Xrow')

    status = "ok"
    err = nothing
    labels = nothing
    K_found = nothing
    noise_pct = nothing
    try
        if algo == "kmeans"
            Random.seed!(seed)
            R = kmeans(X, k; maxiter=300, display=:none, init=:kmpp)
            labels = collect(Int, assignments(R))
            K_found = length(unique(labels))
        elseif algo == "agg"
            D = pairwise(Euclidean(), X; dims=2)
            H = hclust(D; linkage=:complete)
            labels = collect(Int, cutree(H; k=k))
            K_found = length(unique(labels))
        elseif algo == "dbscan"
            R = dbscan(X, eps; min_neighbors=min_samples, min_cluster_size=1)
            raw = collect(Int, assignments(R))
            labels0 = [c == 0 ? -1 : c for c in raw]
            pos = sort(unique(filter(>=(0), labels0)))
            remap = Dict(c => i - 1 for (i, c) in enumerate(pos))
            labels = [c < 0 ? -1 : remap[c] for c in labels0]
            K_found = length(unique(filter(>=(0), labels)))
            noise_pct = 100.0 * count(==(-1), labels) / n
        else
            error("unknown algo $algo")
        end
    catch e
        status = "failed"
        err = sprint(showerror, e)
    end
    result = Dict(
        "status" => status,
        "error" => err,
        "n" => n,
        "d" => d,
        "labels" => labels,
        "K_found" => K_found,
        "noise_pct" => noise_pct,
    )
    open(out_path, "w") do io
        JSON.print(io, result)
    end
    return status
end

jobs = JSON.parsefile(ARGS[1])
println("batch jobs=", length(jobs))
for (i, job) in enumerate(jobs)
    st = run_job(job)
    println(i, "/", length(jobs), " ", get(job, "id", "?"), " ", st)
end
