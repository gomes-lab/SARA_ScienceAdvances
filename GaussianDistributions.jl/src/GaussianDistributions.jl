module GaussianDistributions
# there are fundamental operations with Gaussian distributions:
# 1. marginals
# 2. conditionals
# 3. linear functionals
# 4. sampling
# 5. optimizing hyper-parameters of Gaussian Processes
# this package provides convenient and efficient implementations for these operations
function marginal end
function conditional end

using LinearAlgebra
using LinearAlgebra: checksquare

using Base.Threads
using StaticArrays
using FillArrays
using Statistics
import StatsBase: sample

using CovarianceFunctions
using CovarianceFunctions: gramian, MercerKernel, MultiKernel, AbstractKernel
Statistics.cov(k::MercerKernel, x::AbstractVector) = gramian(k, x)

using ForwardDiff # necessary for value_gradient in util.jl
using DiffResults

export Gaussian, Univariate, Multivariate, VectorGaussian, 𝒩
export mean, cov, var, std, pdf, cdf, nld, nlml
export marginal, conditional, conditional!, sample

using LazyInverse: inverse, Inverse, pseudoinverse
using LazyLinearAlgebra
using LinearAlgebraExtensions: difference, AbstractMatOrFacOrUni, AbstractMatOrFac
using WoodburyIdentity

abstract type AbstractGaussian{M, S} end
Statistics.mean(G::AbstractGaussian) = G.μ
Statistics.cov(G::AbstractGaussian) = G.Σ # ensure Hermitian-ness if Σ is a matrix?
const tol = 1e-6 # default error tolerance for matrix factorizations and samples
const VecOfVec{T} = AbstractVector{<:AbstractVector{T}}

############################# Gaussian Distribution ############################
struct Gaussian{M, S} <: AbstractGaussian{M, S}
    μ::M
    Σ::S
	function Gaussian(μ::Real, σ²::Real)
		σ² ≥ 0 ? new{typeof(μ), typeof(σ²)}(float(μ), σ²) : error("σ² < 0")
	end
	function Gaussian(μ::AbstractVector, Σ::AbstractMatOrFacOrUni)
		length(μ) == checksquare(Σ) || error("dimensions of μ, Σ not compatible")
		new{typeof(μ), typeof(Σ)}(μ, Σ)
	end
	Gaussian(μ, Σ) = new{typeof(μ), typeof(Σ)}(μ, Σ)
end
const 𝒩 = Gaussian
# 1-argument input constructors, assuming mean zero
Gaussian() = Gaussian(0., 1.)
# passing variances only
Gaussian(σ²::Real) = Gaussian(zero(σ²), σ²)
Gaussian(σ²::AbstractVector) = Gaussian(Diagonal(σ²))
Gaussian(Σ::AbstractMatOrFacOrUni) = Gaussian(Zeros(size(Σ, 1)), Σ)
# passing mean and variances
Gaussian(μ::AbstractVector, σ²::Real) = Gaussian(μ, σ²*I(length(μ)))
Gaussian(μ::AbstractVector, σ²::AbstractVector) = Gaussian(μ, Diagonal(σ²))

zero_mean(x) = zero(eltype(x))
# TODO: make this a mutable static vector
zero_mean(x, d::Int) = zeros(eltype(x), d)
zero_mean(d::Int) = x->zero_mean(x, d)

Gaussian(k) = Gaussian(zero_mean, k)

function Base.length(G::Gaussian)
	if G isa Univariate || G isa Multivariate
		return length(G.μ)
	else
		return Inf # assumed to be process
	end
end

const UnivariateGaussian{M, S} = Gaussian{<:Real, <:Real}
const Univariate = UnivariateGaussian
const Uni = Univariate

# (G::Univariate)(i::Int) = marginal(G, i)
Base.getindex(G::Univariate, i) = marginal(G, i) # bracket-indexing

const MultivariateGaussian{M, S} = Gaussian{<:AbstractVector, <:AbstractMatOrFacOrUni}
const Multivariate = MultivariateGaussian
const Multi = Multivariate

# (G::Multivariate)(i::Int) = marginal(G, i)
Base.getindex(G::Multivariate, i) = marginal(G, i) # bracket-indexing

# Gaussian is assumed to be process-valued, if neither Uni, nor Multi
(G::Gaussian)(x) = Gaussian(mean(G)(x), cov(G)(x, x))

############################# convenience ######################################
Statistics.cov(G::Multivariate, i, j) = cov(G)[i, j] # view?
Statistics.var(G::Union{Uni, Multi}) = diag(cov(G))
Statistics.std(G::Union{Uni, Multi}) = sqrt.(max.(var(G), 0))

# variance (computes diagonal of covariance), should be specialized where
# more efficient computation is possible
Statistics.var(G::Gaussian) = x->max(G.Σ(x, x), 0)
Statistics.std(G::Gaussian) = x->sqrt(var(G)(x))
Statistics.mean(G::Gaussian, x::AbstractVector) = marginal(mean(G), x)
# IDEA: allow for randomized approximation of this function akin to "Scaling Gaussian Process Regression with Derivatives"
# Statistics.var(G::Gaussian, x::AbstractVector) = var(G).(x)
function Statistics.var(G::Gaussian, x::AbstractVector)
	σ² = fill(var(G)(x[1]), length(x))
	@threads for i in 2:length(x) # virtually perfect parallel scaling
		σ²[i] = var(G)(x[i])
	end
	return σ²
end
function Statistics.std(G::Gaussian, x::AbstractVector)
	σ = var(G, x)
	@. σ = sqrt(max(σ, 0))
end



Base.copy(G::Gaussian) = Gaussian(copy(G.μ), copy(G.Σ))
function Base.:(==)(G::Gaussian, H::Gaussian)
	return mean(G) == mean(H) && cov(G) == cov(H)
end
function Base.isapprox(G::Gaussian, H::Gaussian; atol = eps(), rtol = eps())
    isapprox(G.μ, H.μ, atol = atol, rtol = rtol) && isapprox(G.Σ, H.Σ, atol = atol, rtol = rtol)
end

function LinearAlgebra.cholesky(G::Multivariate; tol::Real = tol, check::Bool = false) # low rank via cholesky
	cov(G) isa AbstractMatrix || return G # to prevent calling cholesky on a factorization
	S = Symmetric(cov(G))
    C = cholesky(S, Val(true), tol = tol, check = check)
    if check
		issuccess(C) || throw("pivoted cholesky not successful")
	end
    return Gaussian(mean(G), C)
end
function LinearAlgebra.factorize(G::Multivariate; tol::Real = tol)
	cov(G) isa AbstractMatrix || return G # to prevent calling cholesky on a non-factorizable object
	return cholesky(G, tol = tol)
end

include("util.jl")
include("sample.jl")
include("marginal.jl")
include("conditional.jl")
include("multi.jl")
include("linear.jl")
include("density.jl")
include("optimization.jl")
include("plot.jl")

# non-essential extensions
include("inputnoise.jl")

end # GaussianDistributions
