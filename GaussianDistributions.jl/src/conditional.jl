########################## conditional distribution ############################
# TODO: add iterative method conditioning
# TODO: specialize conditional for ConditionalKernel GP
# IDEA: generalize gp conditional with arbitrary linear functional
# IDEA: internally normalize data and covariance before inference for stability
# IDEA have separate type for GP to make conditional type dispatch easier?
# nice syntax
Base.:|(G::AbstractGaussian, t::Tuple) = conditional(G, t...)

# if noise covariance is passed separately, first merge it with the observations
function conditional(G::Gaussian, x, y::AbstractVector, Σ::Union{Real, AbstractMatOrFacOrUni})
    conditional(G, x, 𝒩(y, Σ))
end
# converting vector of noise variances to diagonal covariance matrix via constructor
function conditional(G::Gaussian, x::AbstractVector, y::AbstractVector, σ²::AbstractVector)
    conditional(G, x, 𝒩(y, σ²))
end

# conditional with matrix input X for GPs
function conditional(G::Gaussian, X::AbstractMatrix, y::AbstractVector; tol = tol)
	conditional(G, [c for c in eachcol(X)], y, tol = tol)
end
function conditional(G::Gaussian, X::AbstractMatrix, y::Multivariate; tol = tol)
	conditional(G, [c for c in eachcol(X)], y, tol = tol)
end

############################### Univariate #####################################
conditional(G::Univariate, x::Real) = Gaussian(x, zero(0))
conditional(G::Univariate, x::Real, σ²::Real) = conditional(G, 𝒩(x, σ²))
function conditional(G::Univariate, x::Univariate)
    α = var(G) / (var(G) + var(x))
    μ = (1-α) * mean(G) + α * mean(x)
    σ² = (1-α) * var(G)
    Gaussian(μ, σ²)
end
############################### Multivariate ###################################
conditional(G::Multivariate, i::Int, x::Real) = conditional(G, [i], [x])

# pre-processes index array to be bitarray to allow for "not i" indexing
function conditional(G::Multivariate, i::AbstractVector{<:Integer},
                     Gx::Union{AbstractVector, Multivariate})
    length(i) == length(Gx) || error("number of indices and values in x do not match")
    isdata = [any(==(k), i) for k in eachindex(mean(G))] # TODO: compute in quasi-linear time
    conditional(G, isdata, Gx)
end

# conditional distribution, after knowing that G[i] = x + ε where x + ε ∼ Gx
function conditional(G::Multivariate, i::AbstractVector{<:Bool},
                     Gx::Union{AbstractVector, Multivariate})
    conditional!(copy(G), i, Gx)
end

function conditional!(G::Multivariate, i::AbstractVector{<:Bool},
                      Gx::Union{AbstractVector, Multivariate}; tol::Real = tol)
    any(i) || return G # return if there is nothing to condition on
    j = .!i
    x = Gx isa Gaussian ? mean(Gx) : Gx
    Gi, Gj = marginal(G, i), marginal(G, j)
    Gi = factorize(Gi, tol = tol)
    Σii = Gx isa Gaussian ? factorize(Woodbury(cov(Gx), cov(Gi))) : cov(Gi)
    Λi = inverse(Σii) # pseudo-inverse?
    Σi = @view cov(G)[i, :]

    # assumes G.μ, (G.Σ) contains prior mean (covariance)
    G.μ .+= Σi' * (Λi * difference(x, mean(Gi)))
    G.Σ .-= *(Σi', Λi, Σi) # check if we can pre-allocate memory here. woodbury?

    if x isa AbstractVector # collapse of uncertainty of observed noiseless data
        G.μ[i] .= x
        G.Σ[i, :] .= 0; G.Σ[:, i] .= 0
    end
    return G
end

################################ Process #######################################
# conditional distribution of Gaussian Process
# ASSUMPTION: x and y are vectors of data, not data vectors
# FIXME: does not work if G is a conditional kernel (gives rise to Woodbury K, which does not have factorize(K, tol) signature)
function conditional(G::Gaussian, x::AbstractVector,
                     Gy::Union{AbstractVector, Multivariate}; tol::Real = tol)
    length(x) > 0 || return G # return if an empty vector is passed
	if G.Σ isa ConditionalKernel
		throw("iterative conditioning is not yet supported, call conditional with entire data instead.")
		# TODO: if applicable, use cholesky updating formula, need to keep track of all targets y?
	end
    K = gramian(cov(G), x)
    K = length(x) == 1 ? Matrix(K) : factorize(K, tol = tol)
    y = Gy
    if Gy isa Gaussian # if there is no noise variance, gramian has to be full rank
        y, Σ = mean(Gy), cov(Gy)
        K = length(x) == 1 ? (K + Σ) : factorize(Woodbury(Σ, K)) # TODO allow for iterative inference, like in multi.jl: specialize factorize for LazyMatrixSum
    else
        rank(K) == size(K, 1) || throw(RankDeficientException(1))
    end
    μ = ConditionalMean(G, x, y, K)
    Σ = ConditionalKernel(G, x, K)
    return Gaussian(μ, Σ)
end

######################## conditioning on single datapoints #####################
# IDEA: try to consolidate
# process with 1d input and output
function conditional(G::Gaussian, x::Real, y::Real, σ²::Real)
    conditional(G, [x], [y], σ²)
end
conditional(G::Gaussian, x::Real, y::Real) = conditional(G, [x], [y])

# process with nd input and 1d ouput
function conditional(G::Gaussian, x::AbstractVector{<:Real}, y::Real, σ²::Real)
    conditional(G, [x], [y], σ²)
end
function conditional(G::Gaussian, x::AbstractVector{<:Real}, y::Real)
    conditional(G, [x], [y])
end

############################### Conditional Mean ###############################
# here, T would be the output type ..., or should it be field type?
struct ConditionalMean{M, K, U<:AbstractVector, V<:AbstractVector}
    μ::M
	kxu::K
	u::U
    α::V # α = K \ (y - μ(x)) IDEA: sparsity?
end
ConditionalMean(G::Gaussian, u...) = ConditionalMean(G.μ, G.Σ, G.μ, u...)
function ConditionalMean(μx, kxu, μu, u::AbstractVector, y::AbstractVector, K)
    μ = marginal(μu, u)
    α = K \ difference(y, μ)
    ConditionalMean(μx, kxu, u, α)
end
function (M::ConditionalMean)(x)
	value = M.μ(x)
	if length(M.u) < length(M.α) # this means M.kxu will return a vector
		G = gramian(M.kxu, [x], M.u)
		g = vec(Matrix(G)) # because of BlockFactorization
		value += dot(g, M.α)
	else
	    for i in eachindex(M.α)
	        value += M.kxu(x, M.u[i]) * M.α[i]
	    end
	end
	return value
end
# not possible because of BlockFactorization
# IDEA: don't return it if one dimension is one!
# (M::ConditionalMean)(x) = M.μ(x) + dot(vec(gramian(M.kxu, [x], M.u)), M.α)
# function (M::ConditionalMean)(x) # this doesn't work for vector-valued observations
#     value = M.μ(x)
#     for i in eachindex(M.α)
#         value += M.kxu(x, M.u[i]) * M.α[i]
#     end
#     return value
# end
function marginal(M::ConditionalMean, x::VecOfVec)
	μ = marginal(M.μ, x)
	K = gramian(M.kxu, x, M.u)
	return mul!(μ, K, M.α, 1, 1)
end

########################## Conditional Kernel ##################################
# corresponds to the kernel of the conditional GP, with observations at x
# ASSUMPTION: x is vector of data, even if it is just a single element
struct ConditionalKernel{T, K, KX, KY, U<:AbstractVector,
                    	 S<:AbstractMatOrFac{T}} <: MercerKernel{T}
    kxx::K # prior covariance function of predictions
	kxu::KX # covariance function between observations and predictions
	kuy::KY # covariance function between observations and predictions
    u::U # input data on which we are conditioning
    Kuu::S # covariance between u and u
end
# default to having same covariance between observations and predictions
function ConditionalKernel(k, u::AbstractVector, Kuu::AbstractMatOrFac)
	ConditionalKernel(k, k, k, u, Kuu)
end
function ConditionalKernel(k, u::AbstractVector)
	Kuu = factorize(gramian(k, u))
	ConditionalKernel(k, u, Kuu)
end
function ConditionalKernel(G::Gaussian, u::AbstractVector, Kuu::AbstractMatOrFac)
    ConditionalKernel(cov(G), u, Kuu)
end

# might have to force @nospecialize for K, since recursive Conditional usage
# could be stressing the compiler
# TODO: why lazy evaluation here? for multi-output GPs we have VectorValuedGaussian ...
function (C::ConditionalKernel)(x, y)
    Kxu = gramian(C.kxu, [x], C.u) # no need to evaluate lazily because output dimension is one
    Kuy = (x ≡ y) && (C.kxu ≡ C.kuy) ? Kxu' : gramian(C.kuy, C.u, [y])
	Kxu, Kuy = vec(Matrix(Kuy)), vec(Matrix(Kxu)) # vec(Kxu), vec(Kuy) # necessary because BlockFactorization does not allow vec currently
    C.kxx(x, y) - dot(Kxu, inverse(C.Kuu), Kuy) # bottleneck
end

# for efficiency, gramian should dispatch on the type of the factorization of c
# does it make sense to return a lazy matrix? given each element requires O(n^2) cost ...
import CovarianceFunctions: gramian
function gramian(C::ConditionalKernel, x::AbstractVector, y::AbstractVector)
   Kxu = gramian(C.kxu, x, C.u)
   Kuy = (x ≡ y) && (C.kxu ≡ C.kuy) ? Kxu' : gramian(C.kuy, C.u, y)
   Woodbury(gramian(C.kxx, x, y), Kxu, inverse(C.Kuu), Kuy, -1)
end
# IDEA: could specialize ternary * for Woodbury as middle argument to speed up SoR

################### conditioning on linear operator data #######################
# computes the posterior distribution x | A, b given prior on x and b
# IDEA: the posterior precision matrix is simple: (A'A + cov(x)), exploit this?
conditional(x::Multivariate, A::AbstractMatrix, b::Uni) = _conditional(x, A, b)
conditional(x::Multivariate, A::AbstractMatrix, b::Multivariate) = _conditional(x, A, b)
function _conditional(x::Multivariate, A::AbstractMatrix, b::Union{Uni, Multi})
    ACx = A*cov(x)
    y = difference(mean(b), A*mean(x)) # this allocates anyway, so use mul!?
    Cy = Woodbury(cov(b), A, cov(x), A') # covariance matrix of b
    Cy = factorize(Cy)
    μ = mean(x) + ACx' * (Cy \ y)
    Σ = cov(x) - *(ACx', inverse(Cy), ACx) # Woodbury?
    Gaussian(μ, Σ)
end

# conditioning on noiseless observations
# avoiding type-ambiguity with line 19 by creating "hidden" _conditional
conditional(x::Multivariate, A::AbstractMatrix, b::Real) = _conditional(x, A, b)
conditional(x::Multivariate, A::AbstractMatrix, b::AbstractVector) = _conditional(x, A, b)
function _conditional(x::Multivariate, A::AbstractMatrix, b::Union{Real, AbstractVector})
    μ, Σ = copy(b), zero(cov(x))
    if !issquare(A)
        ACx = A*cov(x)
        y = difference(b, A*mean(x))
        Cy = factorize(ACx * A') # scales cubically in observations *(A, cov(x), A')
        μ = mean(x) + ACx' * (Cy \ y)
        Σ = Symmetric(cov(x) - *(ACx', inverse(Cy), ACx))
    end
    Gaussian(μ, Σ)
end

# convenience for linear functionals parameterized by a
# IDEA: try to consolidate
function conditional(x::Multivariate, a::AbstractVector{<:Real}, b::Real)
    conditional(x, a', b)
end
function conditional(x::Multivariate, a::AbstractVector{<:Real}, b::Uni)
    conditional(x, a', b)
end
