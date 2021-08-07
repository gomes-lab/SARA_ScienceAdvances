####################### kernel for SARA's inner loop ###########################
# lx is length scale for position, respectively
function iSARA(lx::Real, rescale_parameters = nothing)
    kx = Kernel.Lengthscale(Kernel.MaternP(2), lx)
    kx = Kernel.SymmetricKernel(kx)
    if !isnothing(rescale_parameters)
        kx = Kernel.VerticalRescaling(kx, get_rescale(rescale_parameters))
    end
    return kx = kx + Kernel.Line(1.)
end
function iSARA(θ::AbstractVector, rescale_parameters = nothing)
    lx = exp(θ[1])
    iSARA(lx, rescale_parameters)
end

############################# rescaling function ###############################
# the scaling function will take the RGB image gradients directly,
# and convolute them with a window function as follows:
# LSA(x) * (grad(RGB(x)^2 * alpha + 1)
# generalized bell curve
gaussian(μ::Real, σ::Real, β::Real) = x->exp(-(abs(x-μ)/σ)^β/2)

# Parameters is a tuple of gaussian parameters, with (c, μ, σ, β)
function sum_gaussians(x::Real, parameters::Vector{<:NTuple{4, T}}) where {T}
    f = zero(x)
    @inbounds @simd for par in parameters
        f += par[1] * gaussian(par[2], par[3], par[4])(x)
    end
    return f
end

function get_rescale(parameters::Vector{<:NTuple{4, <:Real}})
    (x::Real)->sum_gaussians(x, parameters)
end
get_rescale(p::NTuple{4, <:Real}) = get_rescale([p])

################################################################################
# position is from center of stripe
# optical is a vector of vectors of optical coefficients of position
# i.e. optical[1] is the first coefficient as a function of position
# domain is the minimum and maximum position
# rescale are the parameters of the rescaling function
# sigma is the standard deviation (NOT VARIANCE) of the noise in the coefficients
# grid is a pre-determined grid of potential measurement points
# returns: a vector of points to be measured
function inner(position::Vector{<:Real}, optical::Vector{<:Vector},
                domain::NTuple{2, <:Real} = (-1, 1), σ = .01,
                policy = uncertainty_sampling; # sampling policy
                rescale_parameters = nothing,
                grid::AbstractVector{<:Real} = range(domain[1], domain[2], length = 256))
    # data pre-processing
    x = position
    y = optical
    normalization = [(mean(yi), std(yi)) for yi in y]
    y = center.(y) # does not overwrite input data
    # set up prior Gaussian process
    θ = [.2] # length scale, chosen by cross-validation
    log_θ = log.(θ)
    k = iSARA(log.(θ), rescale_parameters)
    G = Gaussian(k)

    # posterior Gaussian processes
    posteriors = Vector{Gaussian}(undef, length(y))
    @threads for i in eachindex(y)
        m, s = normalization[i]
        posteriors[i] = G | (x, y[i], (σ/s)^2)
        posteriors[i] = s*posteriors[i] + m
    end

    # active learning acquisition
    P = posteriors[1]
    next = policy(P, grid)
    next = [next]
    gradient_uncertainty = integrated_uncertainty(P, grid)
    return next, posteriors, gradient_uncertainty
end
const innerloop = inner

########################## local model evaluation ##############################
function evaluate(G::AbstractVector{<:Gaussian}, x::AbstractVector)
    n = length(G)
    T = eltype(x)
    M = Vector{Vector{T}}(undef, n)
    S = Vector{Vector{T}}(undef, n)
    for (i, P) in enumerate(G)
        M[i], S[i] = evaluate(P, x)
    end
    return M, S
end

function evaluate(G::Gaussian, x::AbstractVector)
    M = marginal(G, x)
    return mean(M), std(M)
end
