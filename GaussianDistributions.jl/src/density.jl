##################### negative logarithm of density ##############################
function nld(G::Univariate, x::Number)
    ((x-mean(G))^2/var(G) + log(var(G)) + log(2π))/2
end

function nld(G::Multivariate, x::AbstractVector)
    x = difference(x, mean(G))
	nld(cov(G), x)
end

function nld(Σ::AbstractMatOrFacOrUni, x::AbstractVector)
    Σ = factorize(Σ) # since factorize on a Inverse is a no-op, saves unnecessary factorization in laplace's method
    Λ = inverse(Σ)
    (dot(x, Λ, x) + logdet(Σ) + length(x) * log(2π))/2
end

# probability density function
pdf(G::Gaussian, x) = exp(-nld(G, x))
pdf(x::Real) = exp(-x^2/2) / sqrt(2π)
function pdf(G::Uni, x::Real)
	σ = sqrt(G.Σ)
	pdf((x - G.μ) / σ) / σ
end

using SpecialFunctions: erf
cdf(x::Real) = (1 + erf(x / sqrt(2))) / 2
cdf(G::Uni, x::Real) = cdf((x - G.μ) / G.Σ)

# lazy
nld(G::Gaussian) = x -> nld(G, x)
pdf(G::Gaussian) = x -> pdf(G, x)
cdf(G::Uni) = x -> cdf(G, x)

entropy(G::Univariate) = log(2π*ℯ*cov(G)) / 2
entropy(G::Multivariate) = (length(G) * log(2π*ℯ) + logdet(cov(G))) / 2

# for Gaussian process
function nld(G::Gaussian, x::AbstractVector, y::AbstractVector)
	nld(marginal(G, x), y)
end
# negative log marginal likelihood
# with noiseless observations y
nlml(K::AbstractMatOrFac, y::AbstractVector) = nld(K, y)
nlml(k, x::AbstractVector, y::AbstractVector) = nlml(gramian(k, x), y)
nlml(G::Gaussian, x::AbstractVector, y::AbstractVector) = nlml(cov(G), x, y)

# with noisy observations y
# if first argument is not Gaussian, assume it's a kernel
function nlml(k, x::AbstractVector, Gy::Multivariate; tol::Real = tol)
	G = gramian(k, x)
	F = factorize(G, tol = tol) # this should be a pivoted cholesky or low rank
	y, Σ = mean(Gy), cov(Gy)
	K = Woodbury(Σ, F)
	nlml(K, y) # reduces to "noiseless" call
end
function nlml(G::Gaussian, x::AbstractVector, Gy::Multivariate; tol::Real = tol)
	if mean(G) != zero_mean # subtract prior mean, if applicable
		mean(Gy) .-= mean(G, x)
	end
	nlml(cov(G), x, Gy)
end
# convenience for passing noise covariance separately
function nlml(k, x::AbstractVector, y::AbstractVector, Σ::Union{Real, AbstractMatOrFacOrUni})
	nlml(k, x, 𝒩(y, Σ))
end

# value and gradient of nlml with noiseless observations
function pushforward(::typeof(nlml), k, x::AbstractVector, y::AbstractVector)
	K = gramian(k, x)
	pushforward(nlml, K, y)
end
# value and gradient of nlml with noisy observations
function pushforward(::typeof(nlml), k, x::AbstractVector, Gy::Multivariate; tol::Real = tol)
	K = gramian(k, x)
	K = factorize(K, tol = tol) # ideally, this is a pivoted cholesky or low rank
	y, Σ = mean(Gy), cov(Gy)
	K = Woodbury(Σ, K)
	pushforward(nlml, K, y)
end
function pushforward(::typeof(nlml), K::AbstractMatOrFac, y::AbstractVector)
	K = factorize(K)
    α = K \ y # coefficients α are the same as in conditional mean, could pre-allocate
	val = (dot(y, α) + logdet(K) + length(y) * log(2π))/2
	K⁻¹ = Matrix(inv(K))
	@. K⁻¹ -= α * α'
	val, (dK::AbstractMatOrFac) -> dot(K⁻¹, dK)/2 # this is O(n^2) per evaluation
end

# could be useful if certain hyper-parameter combinations lead to rank-deficiency
# function safe_nlml(Σ, y)
#     try nlml(Σ, y) catch e;
#         println(e)
#         Inf
#     end
# end

# # TODO: stochastic gradient via stochastic trace estimator and logdet
# # derivative for Multivariate nld
# function pushforward(::typeof(nlml), Σ::AbstractMatOrFac, y::AbstractVector,
# 										dense_inverse::Val{false} = Val(false))
# 	Σ = factorize(Σ)
#     α = Σ \ y # could pre-allocate here for Σ\∇,
# 	val = (dot(y, α) + logdet(Σ) + length(y) * log(2π))/2
#     val, function (dΣ)
# 				(tr(inverse(Σ)*dΣ) - dot(α, dΣ, α))/2 # this is O(n^3) per evaluation
# 			end
# end
