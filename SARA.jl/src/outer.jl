####################### kernel for SARA's outer loop ###########################
# At SLAC in Nov. 2019, used l_T = .03, l_τ = .1 in normalized domain [0, 1]²
# without normalizing, that corresponds to
# l_T = 1400 * 0.03 = 42, and
# l_τ = 2 * 0.1 = .2 (the first number is the range of the true domain of T and log10_τ)
function oSARA(l::AbstractVector)
    k = Kernel.MaternP(2)
    E = Diagonal(inv.(l).^2)
    Kernel.Energetic(k, E)
end

# length scales of temperature, dwelltime, and composition
oSARA(l_T::Real, l_τ::Real, l_c::Real) = oSARA([l_T, l_τ, l_c])
oSARA(l_T::Real, l_τ::Real) = oSARA([l_T, l_τ])

############################### outer loop #####################################
# returns next stripe conditions and conditional GP
function next_stripe(G::Gaussian, x::AbstractArray{<:NTuple}, y::AbstractArray{<:Real},
                     σ²::AbstractArray{<:Real},
                     stripe_conditions::AbstractVector{<:NTuple}, policy)
    return next_stripe(vec(x), vec(y), vec(σ²), stripe_conditions, policy)
end

function next_stripe(G::Gaussian, x::AbstractVector{<:NTuple}, y::AbstractVector{<:Real},
                     σ²::AbstractVector{<:Real},
                     stripe_conditions::AbstractVector{<:NTuple}, policy)
    length(x) == length(y) == length(σ²) || throw(DimensionMismatch("input data has different lengths: $(length(x)), $(length(y)), $(length(σ²))"))
    k_scale = 1/4 # this parameter is the scaling parameter of the kernel, but isn't incorporated in k due to performance bug
    σ² /= k_scale^2
    C = conditional(G, x, Gaussian(y, σ²), tol = 1e-6) # condition GP on data
    return policy(C, stripe_conditions), C
end
