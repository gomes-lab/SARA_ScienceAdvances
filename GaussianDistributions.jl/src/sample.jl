################################# sample #######################################
# IDEA: pre-allocation?
# IDEA: lazy GP sampling?
using LinearAlgebraExtensions: vecofvec # converts a matrix into a vector of vectors
sample(G::Univariate{T}) where T = mean(G) + std(G) * randn(T)

function sample(G::Univariate{T}, k::Int) where T
    x = randn(T, k)
    x .= mean(G) .+ std(G) .* x
end

# sample(G::AbstractGaussian{<:Real}) = sample(G, 1)[]
function sample(G::Multivariate; tol::Real = tol)
    vec(sample(G, 1, tol = tol)[1])
end

# TODO: sample(G, k) should return a vector with k elements, each of which is a sample
# multi-variate Gaussian sample, falls back to cholesky
function sample(G::Multivariate{<:AbstractVector, <:AbstractMatrix}, k::Int; tol::Real = tol)
    C = cholesky(G, tol = tol)
    return sample(C, k, tol = tol)
end
# requires that Σ.L*Σ.L' = Σ of the original multivariate Gaussian
function sample(G::Gaussian{<:AbstractVector, <:Cholesky}, k::Int; tol::Real = tol)
    vecofvec(mean(G) .+ cov(G).L * randn(length(G), k))
end
function sample(G::Gaussian{<:AbstractVector, <:CholeskyPivoted}, k::Int; tol::Real = tol)
    C = cov(G)
    C.tol ≤ tol || throw("CholeskyPivoted does not have sufficient error tolerance: $(C.tol) ≰ $tol")
    L = LowRank(C)
    sample(Gaussian(mean(G), L), k, tol = tol)
end
using LinearAlgebraExtensions: LowRank
function sample(G::Gaussian{<:AbstractVector, <:LowRank}, k::Int; tol::Real = tol)
    L = cov(G)
    L.tol ≤ tol || throw("LowRank does not have sufficient error tolerance: $(L.tol) ≰ $tol")
    L.U ≡ L.V' || throw("sampling with LowRank only supported if U ≡ V'")
    vecofvec(mean(G) .+ L.U * randn(rank(L), k))
end

function sample(G::Gaussian{<:AbstractVector, <:Union{Diagonal, UniformScaling}},
                k::Int; tol::Real = tol)
    vecofvec(mean(G) .+ sqrt(cov(G)) * randn((length(G), k)))
end

function sample(G::Gaussian{<:AbstractVector, <:Woodbury}, k::Int; tol::Real = tol)
    W = cov(G)
    W.U ≡ W.V' || throw("sampling with Woodbury only supported if U ≡ V'")
    s(A) = sample(Gaussian(A), k, tol = tol)
    s(W.A) .+ .*((W.U,), s(W.C)) .+ (mean(G),)
end
