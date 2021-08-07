using OptimizationAlgorithms
const Optimization = OptimizationAlgorithms

# optimization of kernel hyperparameters of Gaussian Process
# could add optimization w.r.t. leave-one-out loss
# TODO: optimization of noise variance
# IDEA: pass inference method (Cholesky, Iterative, Stochastic, ...)
optimize(k, θ, x, y, σ²::Real) = optimize(k, θ, x, 𝒩(y, σ²))
function optimize(k, θ::AbstractVector, x::AbstractVector,
                  y::Union{AbstractVector, Multivariate})
    optimize!(k, copy(θ), x, y)
end
optimize!(k, θ, x, y, σ²::Real) = optimize!(k, θ, x, 𝒩(y, σ²))
function optimize!(k, θ::AbstractVector, x::AbstractVector,
                   y::Union{AbstractVector, Multivariate})
    val(θ) = nlml(k(θ), x, y) # make this "safe"?
    function valdir(θ)
        v, g = value_gradient(nlml, k, θ, x, y) # need to add tol if y is Multivariate
        g .*= -1
        v, g
    end
    # optimization based on L-BFGS with m stages
    m = 5
    D = Optimization.CustomDirection(val, valdir, θ)
    D = Optimization.LBFGS(D, θ, m, check = false)
    # scaling proposed Dection to unit length before doing line search,
    # avoids jumping to sub-optimal and very large parameter vectors
    D = Optimization.UnitDirection(D)
    # to accept a set, do line search for step size to ensure decrease
    D = Optimization.DecreasingStep(D, θ)
    # finally, find fixed point of the iterations
    _, t = Optimization.fixedpoint!(D, θ)
    return θ
end

# here, k(θ) is assumed to be a kernel function
# computes both value and gradient
function value_gradient(::typeof(nlml), k, θ::AbstractVector, x::AbstractVector,
                        y::Union{AbstractVector, Multivariate})
    val, push = pushforward(nlml, k(θ), x, y)

    ∇ = (x, y) -> reshape(ForwardDiff.gradient(z->k(z)(x, y), θ), :, 1)
    G = gramian(∇, x) # lazily evaluated matrix of kernel gradients
    G_matofvec = G.A
    G = matvec2vecmat(G_matofvec) # casts G to a vector of dense matrices. IDEA: make lazy!

    grad = similar(θ)
    @threads for i in eachindex(grad) # parallelizing here is efficient
        grad[i] = push(G[i])
    end
    return val, grad
end

function gradient(::typeof(nlml), k, θ::AbstractVector,
                    x::AbstractVector, y::Union{AbstractVector, Multivariate})
    value_gradient(nlml, k, θ, x, y)[2]
end
using CovarianceFunctions: Gramian
matvec2vecmat(A::Gramian{<:AbstractVecOrMat}) = matvec2vecmat(Matrix(A))
# convert matrix of vectors to vector of matrices
function matvec2vecmat(A::AbstractMatrix{<:AbstractVecOrMat})
    k = length(A[1])
    all(==(k), (length(Ai) for Ai in A)) || throw(DimensionMismatch("component vectors do not have the same size"))
    n, m = size(A)
    B = Vector{Matrix{eltype(A[1])}}(undef, k)
    for i in eachindex(B) # this is still inefficient, because it calculates the gradient multiple times
        B[i] = [a[i] for a in A]
    end
    return B
end

########################### optimizing noise variance ##########################
function value_gradient(::typeof(nlml), log_σ²::Real, K::Factorization, y::AbstractVector)
    n = length(y)
    Σ = exp(log_σ²)*I(n)
    W = Woodbury(Σ, K)
    val, push = pushforward(nlml, W, y)
    grad = push(Σ)
    return val, grad
end

function optimize(σ²::Real, k, x::AbstractVector, y::AbstractVector; tol::Real = tol)
    # pre-factorize kernel matrix
    n = length(x)
    K = gramian(k, x)
    K = factorize(K, tol = tol)
    function val(log_σ²::Real)
        Σ = exp(log_σ²)*I(n)
        nlml(Woodbury(Σ, K), y)
    end
    val(θ::AbstractVector) = val(θ[1])
    function valdir(log_σ²::Real)
        v, g = value_gradient(nlml, log_σ², K, y)
        g = -g
        v, g
    end
    function valdir(θ::AbstractVector)
        v, g = valdir(θ[1])
        v, [g]
    end
    θ = [log(σ²)]
    D = Optimization.CustomDirection(val, valdir, θ)
    D = Optimization.LBFGS(D, θ, 2, check = false) # does this reduce to secant?
    D = Optimization.UnitDirection(D)
    D = Optimization.DecreasingStep(D, θ)
    Optimization.fixedpoint!(D, θ)
    return σ² = exp(θ[1])
end
