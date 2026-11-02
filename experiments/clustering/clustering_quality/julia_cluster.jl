#!/usr/bin/env julia
using Pkg
Pkg.activate(@__DIR__)
using Clustering
using Distances
using JSON
using Random
using DelimitedFiles

function get_arg(args, name, default=nothing)
    i = findfirst(==(name), args)
    i === nothing && return default
    return args[i + 1]
end

function main()
    args = ARGS
    algo = get_arg(args, "--algo")
    x_path = get_arg(args, "--X")
    out_path = get_arg(args, "--out")
    k = parse(Int, get_arg(args, "--k", "0"))
    seed = parse(Int, get_arg(args, "--seed", "0"))
    eps = parse(Float64, get_arg(args, "--eps", "0.3"))
    min_samples = parse(Int, get_arg(args, "--min_samples", "5"))

    Xrow = readdlm(x_path, ',', Float64)   # n × d
    n, d = size(Xrow)
    X = Matrix(Xrow')                      # d × n

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
            # Clustering.jl uses 0 for noise
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
        labels = nothing
    end

    result = Dict(
        "status" => status,
        "error" => err,
        "n" => n,
        "d" => d,
        "labels" => labels,
        "K_found" => K_found,
        "noise_pct" => noise_pct,
        "orientation_note" => "CSV rows=observations; transposed to d×n for Clustering.jl; labels in CSV row order",
    )
    open(out_path, "w") do io
        JSON.print(io, result, 2)
    end
    println(status, " ", algo, " n=", n)
end

main()
