# using Plots
using RecipesBase
# IDEA: use MAKIE for faster compile?
# IDEA: 2d and 3d recipies
# default range
plotting_range(G::Gaussian) = range(-1, 1, length = 256)
function plotting_range(G::Gaussian{<:ConditionalMean})
	range(minimum(G.μ.x) - std(G.μ.x), maximum(G.μ.x) + std(G.μ.x), length = 256)
end

@recipe function f(G::Gaussian, x::AbstractVector = plotting_range(G))
	seriestype  :=  :path
	μ = mean(G, x)
	σ = std(G).(x)
	ribbon := 2σ
	(x, μ)
end

# IDEA: add a few samples to the plot (currently requires a dense factorization)
# if n > 0
# 	s = sample(f, n, tol = 1e-6)
# 	plot!(xs, s, seriesalpha = .7)
# end
