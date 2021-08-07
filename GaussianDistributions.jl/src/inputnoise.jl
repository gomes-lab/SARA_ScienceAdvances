########################## NIGP (Noisy Input GP) ###############################
# IDEA: training procedure for input variances (chain derivatives through corrective term)
# Laplace approximation to treating measurement points as multivariate Gaussians?

# conditional with two covariance inputs inspired by approximation put forth in
# "Gaussian Process Training with Input Noise" by McHutchon and Rasmussen 2011
# generalizes the work by allowing non-constant input noise variances
# ASSUMPTION: x and y are vectors of data, not data vectors
function conditional(G::Gaussian, Gx::Gaussian{<:AbstractVector, <:Diagonal},
                                  Gy::Gaussian{<:AbstractVector, <:Diagonal})
    x, y = mean(Gx), mean(Gy)
    Σx::Diagonal, Σy::Diagonal = cov(Gx), cov(Gy)
    C = conditional(G, x, y, Σy) # canonical conditioning
    ∇(x::Real) = ForwardDiff.derivative(mean(C), x) # differentiate posterior mean
    ∇(x::AbstractVector) = ForwardDiff.gradient(mean(C), x)
    Σ = Diagonal(zeros(length(x)))
    for i in eachindex(x)
        ∇x = ∇(x[i])
        Σ[i, i] = max(dot(∇x, Σx[i, i], ∇x), 0)
    end
    Σ += Σy
    conditional(G, x, y, Σ)
end

# WARNING: Σx is the d by d noise covariance for all inputs x,
# similar to σy² being the noise variance for all outputs y
function conditional(G::Gaussian,
                     x::AbstractVector, Σx::Union{Real, AbstractMatrix},
                     y::AbstractVector, σy²::Real)
    n = length(x)
    Σx = Diagonal(fill(Σx, n))
    Gx = Gaussian(x, Σx)
    Σy = σy² * I(n)
    Gy = Gaussian(y, Σy)
    conditional(G, Gx, Gy)
end
