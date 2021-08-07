# specializations for multi-output GPs
# names: MultiOuputGaussian, MultiOutputProcess, VectorGaussian, VectorGaussianProcess
struct VectorGaussian{M, S} <: AbstractGaussian{M, S}
    μ::M
    Σ::S
	d::Int # output dimension
end
VectorGaussian(k, d::Int) = VectorGaussian(zero_mean(d), k, d)

(G::VectorGaussian)(x) = Gaussian(mean(G)(x), cov(G)(x, x))

Statistics.var(G::VectorGaussian) = x->max.(diag(G.Σ(x, x)), 0)
Statistics.std(G::VectorGaussian) = x->sqrt.(var(G)(x))
Statistics.mean(G::VectorGaussian, x::AbstractVector) = marginal(mean(G), x, G.d)
# TODO: call var(G).(x) for efficiency?
# Statistics.var(G::VectorGaussian, x::AbstractVector) = max.(diag(gramian(cov(G), x)), 0)
# # function var(k::MultiKernel, x::AbstractVector{<:Real}) end
# Statistics.std(G::VectorGaussian, x::AbstractVector) = sqrt.(var(G, x))

# getindex for multi-output GP selects specific ouput dimenion
function Base.getindex(G::VectorGaussian, i)
	μ(x) = mean(G)(x)[i]
	Σ(x, y) = Σ isa MultiKernel ? cov(G)[i, i](x, y) : cov(G)(x, y)[i, i]
	Gaussian(μ, Σ)
end

############################ vector conditional ################################
# iterative_trait(::AbstractMatOrFac) = Val(false)
# iterative_trait(::BlockFactorization) = Val(true)
function conditional(G::VectorGaussian, x::AbstractVector,
                     Gy::Union{AbstractVector{<:Number}, Multivariate}; tol::Real = tol)
    length(x) > 0 || return G # return if an empty vector is passed
    K = gramian(cov(G), x)
    K = length(x) == 1 ? Matrix(K) : factorize(K, tol = tol)
    y = Gy
    if Gy isa Gaussian
        y = mean(Gy)
        K = K isa Matrix ? K + cov(Gy) : LazyMatrixSum(K, cov(Gy)) # add noise variance to kernel matrix
    end
	d = length(Gy) ÷ length(x)
    μ = VectorConditionalMean(G, x, y, d, K)
    Σ = VectorConditionalKernel(G, x, K)
    return VectorGaussian(μ, Σ, d)
end

# conditional helpers
function conditional(G::VectorGaussian, x, y::Union{AbstractMatrix, VecOfVec}; tol::Real = tol)
	conditional(G, x, _vec(y); tol = tol)
end

# if Y is passed as a matrix, Σ is interpreted as a matricified version of the corresponding diagonal covariance
function conditional(G::VectorGaussian, x, y::MT, Σ::MT; tol::Real = tol) where {MT <: Union{AbstractMatrix, VecOfVec}}
	Gy = Gaussian(_vec(y), Diagonal(_vec(Σ)))
	conditional(G, x, Gy, tol = tol)
end

function conditional(G::VectorGaussian, x, y::Union{AbstractMatrix, AbstractVector}, σ²::Real; tol::Real = tol)
	Gy = Gaussian(_vec(y), σ²)
	conditional(G, x, Gy; tol = tol)
end

function conditional(G::VectorGaussian, x, y::AbstractVector{<:Number}, Σ; tol = tol)
	conditional(G, x, Gaussian(y, Σ), tol = tol)
end

# conditional with matrix input X
# unnecessary if we make first argument AbstractGaussian in main conditional file
function conditional(G::VectorGaussian, X::AbstractMatrix, y::Union{AbstractVector, Multivariate}; tol = tol)
	conditional(G, [x for x in eachcol(X)], y, tol = tol)
end

############################### Conditional Mean ###############################
# here, T would be the output type ..., or should it be field type?
struct VectorConditionalMean{M, K, U<:AbstractVector, V<:AbstractVector}
	μ::M # prior mean of predictions
	kxu::K # covariance between prediction and observation points (not necessarily prior cov for multi-GP)
    u::U
    α::V # α = K \ (y - μ(x)) IDEA: sparsity?
	d::Int
end
# converting Gaussian to separate mean and covariance function
VectorConditionalMean(G::VectorGaussian, x...) = VectorConditionalMean(G.μ, G.Σ, G.μ, x...)
# μx is mean prediction
# μu is mean observation
# kxu is covariance between predictions and observations
# y are observation at locations u
# Kuu is the factorized covariance matrix between observations
function VectorConditionalMean(μx, kxu, μu, u::AbstractVector, y::AbstractVector{<:Real}, d::Int, Kuu)
    μ = marginal(μu, u, d)
    α = Kuu \ difference(y, μ)
    VectorConditionalMean(μx, kxu, u, α, d)
end
(M::VectorConditionalMean)(x) = M.μ(x) + gramian(M.kxu, [x], M.u) * M.α

function Base.getindex(M::VectorConditionalMean, i::Int)
	kxu(x, y) = Matrix(M.kxu(x, y))[i, :] # covariance of ith predictive with all observation dimensions
	# kxu(x, y) = M.kxu(x, y)[i, :] # possible without Matrix?
	mean_component(x) = M.μ(x)[i, :] + gramian(kxu, [x], M.u) * M.α
end
# IDEA, but change-intensive to implement
# function Base.getindex(M::VectorConditionalMean, i::Int)
# 	k(x, y) = M.kxu(x, y)[i, :] # covariance of ith predictive with all observation dimensions
# 	ConditionalMean(M.μ, k, M.x, M.α)
# end
function marginal(M::VectorConditionalMean, x::AbstractVector)
    μ = marginal(M.μ, x, M.d)
    K = gramian(M.kxu, x, M.u)
    return mul!(μ, K, M.α, 1, 1)
end

########################## Conditional Kernel ##################################
# corresponds to the kernel of the conditional GP, with observations at x
# ASSUMPTION: x is vector of data, even if it is just a single element
struct VectorConditionalKernel{T, K, KX, U<:AbstractVector,
                    		   S<:AbstractMatOrFac{T}} <: MultiKernel{T}
    kxx::K # prior covariance function of predictions
	kxu::KX # covariance function between observations and predictions
	# kuy::KY # TODO: covariance function between observations and predictions
    u::U # input data on which we are conditioning
    Kuu::S # covariance between x and x
end
# default to having same covariance between observations and predictions
function VectorConditionalKernel(k, u::AbstractVector, Kuu::AbstractMatOrFac)
	VectorConditionalKernel(k, k, u, Kuu)
end

# Kxx might be constructed based on kernel type
function VectorConditionalKernel(k::MultiKernel, u::AbstractVector)
    Kuu = factorize(gramian(k, u))
    VectorConditionalKernel(k, u, Kuu)
end
function VectorConditionalKernel(G::VectorGaussian, u::AbstractVector, Kuu::AbstractMatOrFac)
    VectorConditionalKernel(cov(G), u, Kuu)
end

function (C::VectorConditionalKernel)(x::AbstractVector, y::AbstractVector)
    Kxu = gramian(C.kxu, [x], C.u) # why Matrix?
    Kyu = (x ≡ y) ? Kxu : gramian(C.kxu, [y], C.u)
	K = Matrix(C.kxx(x, y)) # need to convert this to Matrix to allocate, or do Woodbury
	K .-= Kxu * (C.Kuu \ Matrix(Kyu)') # (Kux', inverse(C.Kxx), Kuy)
	# Woodbury(C.kxx(x, y), Kxu, inverse(C.Kuu), Kyu') # inverse should already work
end

Base.getindex(C::VectorConditionalKernel, i, j) = (x, y)->C(x, y, i, j)
# only calculates output variance of single component
function (C::VectorConditionalKernel)(x::AbstractVector, y::AbstractVector, i, j)
	Kux = Matrix(gramian(C.kxu, [x], C.u))'[:, i] # can we do this without matrix?
	Kuy = (x ≡ y) && (i == j) ? Kux : Matrix(gramian(C.kxu, [y], C.u))'[:, j]
	K = C.kxx(x, y)[i, j]
	M = Kux' * (C.Kuu \ Kuy)
	K .- M # *(Kux', inverse(C.Kxx), Kuy)
end

# for efficiency, gramian should dispatch on the type of the factorization of c
# does it make sense to return a lazy matrix? given each element requires O(n^2) cost ...
import CovarianceFunctions: gramian
function gramian(C::VectorConditionalKernel, x::AbstractVector, y::AbstractVector)
   Kxu = gramian(C.kxu, x, C.u)
   Kyu = (x ≡ y) ? Kxu : gramian(C.kxu, y, C.u)
   Woodbury(gramian(C.kxx, x, y), Kux', inverse(C.Kxx), Kuy, -1)
end

# IDEA: take inspiration from finite-dimensional linear operator conditonal in conditiona.jl
# # # global optimization via deconvolution
# function conditional(G::Gaussian, H::VectorGaussian, X::AbstractMatrix, y::AbstractVector)
# 	k, h = cov(G), cov(H)
#     s = k + h # sum of kernels corresponds to convolution of GPs
#     G = gramian(s, X)
#     F = factorize(G)
#     α = F \ y
#     x = [c for c in eachcol(X)]
# 	VectorConditionalMean()
# 	VectorConditionalKernel()
# end

# doodles
################################################################################
# constructing multi-output GP with separable covariance kernel
# function Base.:*(A::Union{Real, AbstractMatOrFacOrUni}, G::Gaussian)
#     μ = x->A*mean(G, x)
#     Σ = Kernel.Separable(A, k)
#     Gaussian(μ, Σ)
# end
################################################################################

# multi-output process with nd input and nd ouput
# this doesn't work for conditional kernel
# function conditional(G::Gaussian{<:Any, <:MultiKernel}, x::AbstractVector{<:Real}, y::AbstractVector{<:Real},
#     Σ = nothing, ::Val{true} = ismulti(cov(G)); tol::Real = 1e-12)
#     conditional(G, [x], [y], Σ, tol = tol)
# end

# if we don't rely on Kernel.jl, this is still a bit of a headache
# function conditional(G::Gaussian, x::AbstractVector{<:Real}, y::AbstractVector{<:Real},
#     Σ = nothing, ::Val{true} = ismulti(cov(G)); tol::Real = 1e-12)
#     conditional(G, [x], [y], Σ, tol = tol)
# end
# function ismulti end
# ismulti(::AbstractKernel) = Val(false)
# ismulti(::MultiKernel) = Val(true)
# ismulti(k::Any) = error("to enable single-datapoint conditioning with a multi-ouput process, please define ismulti(::$(typeof(k)) = Val(true)")
# ismulti(::ConditionalKernel{K}) where K = ismulti(K)
