################################ marginal ######################################
function marginal end
marginal(G::Univariate, i) = i == 1 ? G : throw(DimensionMismatch("Univariate Gaussian does not have index $i"))
marginal(G::Multivariate, i) = Gaussian(G.μ[i], G.Σ[i, i])
marginal(G::Multivariate, i::AbstractVector) = Gaussian(G.μ[i], G.Σ[i, i])
# IDEA: could pass view::Val{true} in marginal
marginalview(G::Multivariate, i) = @views Gaussian(G.μ[i], G.Σ[i, i])
marginalview(G::Multivariate, i::AbstractVector) = @views Gaussian(G.μ[i], G.Σ[i, i])

# WARNING: expects x to be vector of data
function marginal(G::Gaussian, x::AbstractVector)
	μ = marginal(mean(G), x) # evaluation of the mean could be lazy
	Σ = gramian(cov(G), x)
	return Gaussian(μ, Σ)
end
marginal(k::MercerKernel, x::AbstractVector) = gramian(k, x)
# fallback for GP mean functions
# marginal(f, x::AbstractVector) = f.(x)
# this works for vector functions
# IDEA: could this be made better by just working with static arrays as input
# and then vectorizing them? seems feasible
# marginal(f, x::StaticVector) = vec(f.(x)) # output needs to be static array
function marginal(f, x::AbstractVector)
	fx = f(x[1])
	d, T = length(fx), eltype(fx)
	marginal(f, x, d, T)
end
function marginal(f, x::AbstractVector, d::Int, T::DataType = eltype(f(x[1])))
	fx = zeros(T, (d, length(x)))
	for i in eachindex(x)
		fx[:, i] .= f(x[i])
	end
	return vec(fx)
end
