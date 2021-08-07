# extending linear operations for Gaussian types
function Base.:*(a::Real, G::Union{Univariate, Multivariate})
    return Gaussian(a*mean(G), a^2*cov(G))
end
function Base.:*(A::AbstractMatOrFacOrUni, G::Multivariate)
    if !(size(A, 2) == length(G) || size(A, 2) == 1)
        throw(DimensionMismatch("size of G = $(length(G)) inconsistent with A = $(size(A))"))
    end
    μ = A*mean(G)
    Σ = *(A, cov(G), A')
    return Gaussian(μ, Σ)
end
# vector * scalar distribution gives rise to rank-one covariance
Base.:*(a::AbstractVector, G::Uni) = Gaussian(a*mean(G), LowRank(a*std(G)))

Base.:/(G::Union{Univariate, Multivariate}, a::Real) = (1/a)*G
Base.:\(a::Real, G::Union{Univariate, Multivariate}) = (1/a)*G
function Base.:\(A::AbstractMatOrFacOrUni, G::Multivariate)
    F = factorize(A)
    μ = F \ mean(G)
    Σ = (A\(A\cov(G))')'
    Gaussian(μ, Σ)
end
Base.:+(G::Univariate, b::Real) = Gaussian(mean(G) + b, cov(G))
Base.:-(G::Univariate, b::Real) = Gaussian(mean(G)-b, cov(G))

Base.:+(G::Multivariate, b::AbstractVector) = Gaussian(mean(G) + b, cov(G))
Base.:-(G::Multivariate, b::AbstractVector) = Gaussian(mean(G)-b, cov(G))

Base.:-(G::Union{Univariate, Multivariate}) = Gaussian(-mean(G), cov(G))

function Base.sum(G::Multivariate)
    μ = sum(mean(G))
    Σ = sum(cov(G))
    Gaussian(μ, Σ)
end
function LinearAlgebra.dot(G::Multivariate, x::AbstractVector)
    μ = dot(mean(G), x)
    Σ = dot(x, cov(G), x)
    Gaussian(μ, Σ)
end
LinearAlgebra.dot(x::AbstractVector, G::Multivariate) = dot(G, x)
function Base.diff(G::Multivariate)
    μ = diff(mean(G))
    Σ = diff(diff(cov(G), dims = 1), dims = 2)
    Gaussian(μ, Σ)
end

################################# Process ######################################
# assumes f is a deterministic function
# IDEA: could specialize mean to the closures μ to speed up batch computation
function Base.:*(f, G::Gaussian)
    μ(x) = f(x) * mean(G)(x)
    Σ(x, y) = f(x) * cov(G)(x, y) * f(y)
    Gaussian(μ, Σ)
end
function Base.:+(G::Gaussian, f)
    μ(x) = mean(G)(x) + f(x)
    Σ(x, y) = cov(G)(x, y)
    Gaussian(μ, Σ)
end
Base.:*(a::Real, G::Gaussian) = (x->a) * G
# Base.:*(G::Gaussian, a::Real) = a * G

Base.:/(G::Gaussian, a::Real) = (x->inv(a)) * G

Base.:+(G::Gaussian, b::Real) = G + (x->b)
Base.:+(b::Real, G::Gaussian) = G + b

Base.:-(G::Gaussian) = (x -> -1) * G
Base.:-(G::Gaussian, b::Real) = G + (x -> -b)
# Base.:-(b::Real, G::Gaussian) = b + -G

# other argument positioning
Base.:*(G::Gaussian, a) = a * G
Base.:+(a, G::Gaussian) = G + a
Base.:\(a, G::Gaussian) = G/a
Base.:-(b, G::Gaussian) = b + -G

######################## input transformations #################################
# IDEA: output_transformation defaulting to laplace approximation
function input_transformation(G::Gaussian, f)
    μ(x) = G.μ(f(x))
    Σ(x, y) = G.Σ(f(x), f(y))
    Gaussian(μ, Σ)
end
const warp = input_transformation # from warped Gaussian processes
Base.:∘(G::Gaussian, f) = warp(G, f)

# a.k.a. translate
shift(G::Gaussian, b::Real) = G ∘ (x->x+b)
scale(G::Gaussian, a::Real) = G ∘ (x->a*x)

################## integral and differential functionals #######################
function integral end
function gradient end
function value_gradient end
# TODO:
function hessian end
function value_gradient_hessian end

# TODO: get rid of dimensionality d
# TODO: add noise variance for better conditioning
function conditional(G::Gaussian, x::VecOfVec, ::typeof(gradient),
                    y::AbstractVector, d::Int; tol = tol)
    y = _vec(y)
    H = gradient(G, d) # make gradient process for observations
    Kxx = factorize(gramian(H.Σ, x)) # add a noise variance?
    # Kxx = LazyMatrixSum(σ²*I(d*n), factorize(gramian(H.Σ, x))) # add a noise variance?
    kxu(x, u) = reshape(ForwardDiff.gradient(z->G.Σ(x, z), u), 1, :)
    kuy(u, y) = reshape(ForwardDiff.gradient(z->G.Σ(z, y), u), :, 1)
    μ = ConditionalMean(G.μ, kxu, H.μ, x, y, Kxx)
    Σ = ConditionalKernel(G.Σ, kxu, kuy, x, Kxx)
    return Gaussian(μ, Σ)
end

function conditional(G::Gaussian, x::VecOfVec, ::typeof(value_gradient),
                     y::AbstractVector, d::Int; tol = tol)
    y = _vec(y)
    H = value_gradient(G, d) # make gradient process for observations
    Kxx = factorize(gramian(H.Σ, x))
    kxu(x, u) = reshape(value_gradient(z->G.Σ(x, z), u), 1, :)
    kuy(u, y) = reshape(value_gradient(z->G.Σ(z, y), u), :, 1)
    μ = ConditionalMean(G.μ, kxu, H.μ, x, y, Kxx)
    Σ = ConditionalKernel(G.Σ, kxu, kuy, x, Kxx)
    return Gaussian(μ, Σ)
end

# general design question: should these functions return joint distribution
# or only marginal distribution of functional?
# could be solved with "multi" or "joint" argument
# integral of 1d GP
# should we instead extend sum?
function integral(G::Gaussian, a::Real, b::Real, n::Int = 128)
    x, w = range(a, b, length = n), Fill(1/n, n) # quadrature nodes and weights
    Gx = marginal(G, x)
    dot(w, Gx)
end
const ∫ = integral

function gradient(G::Gaussian, d::Int)
    μ(x::AbstractVector) = ForwardDiff.gradient(mean(G), x)
    Σ = CovarianceFunctions.GradientKernel(cov(G))
    VectorGaussian(μ, Σ, d)
end
const ∇ = gradient

# computes joint distribution of function and its gradient
# d is the dimensionality of the input space of the function
function value_gradient(G::Gaussian, d::Int)
    μ(x::AbstractVector) = value_gradient(mean(G), x)
    Σ = CovarianceFunctions.ValueGradientKernel(cov(G))
    VectorGaussian(μ, Σ, d+1)
end

function hessian(G::Gaussian, d::Int)
    μ(x::AbstractVector) = vec(ForwardDiff.hessian(mean(G), x))
    Σ = CovarianceFunctions.HessianKernel(cov(G))
    VectorGaussian(μ, Σ, d^2)
end

function value_gradient_hessian(G::Gaussian, d::Int)
    μ(x::AbstractVector) = value_gradient_hessian(mean(G), x)
    Σ = CovarianceFunctions.ValueGradientHessianKernel(cov(G))
    VectorGaussian(μ, Σ, d^2 + d + 1)
end

function value_derivative(G::Gaussian)
    μ(x::Real) = value_derivative(mean(G), x)
    Σ = CovarianceFunctions.ValueDerivativeKernel(cov(G))
    VectorGaussian(μ, Σ, 2)
end
function derivative(G::Gaussian)
    μ(x::Real) = ForwardDiff.derivative(mean(G), x)
    Σ = CovarianceFunctions.DerivativeKernel(cov(G))
    Gaussian(μ, Σ)
end
