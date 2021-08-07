using Pkg

ssh = true
if ssh
    git = "git@github.com:SebastianAment/"
else
    git = "https://github.com/SebastianAment/"
end

add(x) = Pkg.add(Pkg.PackageSpec(url = git * x * ".git"))

add("LazyInverse.jl")
add("LinearAlgebraExtensions.jl")
add("KroneckerProducts.jl")
add("WoodburyIdentity.jl")
add("OptimizationAlgorithms.jl")
add("CovarianceFunctions.jl")

# not reachable without ssh
if ssh
    add("GaussianDistributions.jl")
end
