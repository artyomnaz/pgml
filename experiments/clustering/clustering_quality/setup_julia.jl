using Pkg
Pkg.activate(@__DIR__)
Pkg.add(["Clustering", "CSV", "DataFrames", "JSON", "Distances"])
using Clustering
println("Clustering installed OK")
