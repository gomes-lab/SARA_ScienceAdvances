###### conversion from optical stripe data to global optical gradient data #####
# 1. convert data to global data using inner loop data with iSARA kernel
# using temp_profile_si
# l is lengthscale, θ are rescaling parameters of iSARA kernel
# nout is number of outer temperatures to record gradients at
# T_fraction is lowest temperature as a fraction of T_max
# k is kernel used for GP inference
# only for 2d data, and backward compatability
function stripe_to_global(x::AbstractVector, y::AbstractVector{<:AbstractVector},
                          σ::Real, k, P::TemperatureProfile, T_max::Real, log10_τ::Real,
                          relevant_T, input_noise::Union{Val{true}, Val{false}} = Val(true))
    stripe_to_global(x, y, σ, k, P, (T_max, log10_τ), relevant_T, input_noise)[2:3]
end

# condition is tuple of (T_max, τ, composition, ...) conditions
function stripe_to_global(x::AbstractVector, y::AbstractVector{<:AbstractVector},
                        σ::Real, k, P::TemperatureProfile, condition::NTuple,
                        relevant_T, input_noise::Union{Val{true}, Val{false}} = Val(true))
    all(==(length(x)), length.(y)) || throw(DimensionMismatch("x and elements of y do not have same lengths: length(x) = $(length(x)) and length.(y) = $(length.(y))"))
    T_max, log10_τ = condition[1:2]
    other_conditions = condition[3:end]
    G = Gaussian(k) # define prior GP
    temperature_processes = Vector{Gaussian}([G for _ in 1:length(y)])
    @threads for j in eachindex(y)
        C = get_temperature_process(G, x, y[j], σ, P, T_max, log10_τ, input_noise)
        temperature_processes[j] = C
    end
    C = temperature_processes[1]
    Tout = relevant_T(T_max, log10_τ) # IDEA: could add composition as dimension
    conditions = tuple.(Tout, log10_τ, other_conditions...)
    gradients, uncertainty = get_global_data(temperature_processes, Tout)
    return conditions, gradients, uncertainty
end

# converts stripe of xrd measurements to integrated gradients
# x is position
# Y is integrated XRD image
function xrd_to_global(x::AbstractVector, Y::AbstractMatrix, σ, k,
                       P::TemperatureProfile, condition::NTuple, relevant_T,
                       input_noise::Union{Val{true}, Val{false}} = Val(true); rtol::Real = 1e-2)
    y = svd_coefficients(Y, rtol) # get coefficients of Y in SVD basis up to relative tolerance 1e-2
    stripe_to_global(x, y, σ, k, P, condition, relevant_T, input_noise)
end

# uses data optical data as a function of position x and the inverse
# temperature profile to compute a Gaussian process as a function of temperature
# G is prior process of position, σ is noise standard deviation
# x is vector of positions corresponding to reflectance measurements y
function get_temperature_process(G::Gaussian, x::AbstractVector,
                                 y::AbstractVector, σ::Real,
                                 P::TemperatureProfile, T_max::Real, log10_τ::Real,
                                 input_noise::Val{false})
    C = conditional(G, x, Gaussian(y, σ^2), tol = 1e-6) # condition Gaussian process in position
    inv_profile = inverse_profile(P, T_max, log10_τ)
    input_transformation(C, inv_profile) # transform the input of conditional process
end

# same as above but takes into account the uncertainty in the temperature
function get_temperature_process(G::Gaussian, x::AbstractVector,
                                 y::AbstractVector, σ_out::Real,
                                 P::TemperatureProfile, T_max::Real, log10_τ::Real,
                                 input_noise::Val{true})
    # first, get regular GP w.r.t. T
    C = get_temperature_process(G, x, y, σ_out, P, T_max, log10_τ, Val(false))
    # with input noise, adjust GP with generalized NIGP model
    T = profile(P, T_max, log10_τ).(x) # maps position of measured optical data to temperature
    d = (t->ForwardDiff.derivative(mean(C), t)).(T)
    var_in = temperature_uncertainty(P, T_max, log10_τ).(x) # input uncertainty corresponding to positions
    Σ = @. var_in * d^2 + σ_out^2
    Σ = Diagonal(Σ)
    C = conditional(G, x, Gaussian(y, Σ), tol = 1e-6) # calculate new conditional process with adjusted noise variance
    inv_profile = inverse_profile(P, T_max, log10_τ)
    input_transformation(C, inv_profile) # transform the input to temperature domain
end

# converts Gaussian processes of temperature with optical target data
# to integrated gradient information of the processes at the given outer_temperature
function get_global_data(temperature_processes::AbstractVector{<:Gaussian},
                         outer_temperature::AbstractVector{<:Real},
                         temperature_domain::NTuple{2, <:Real} = (0, 1400))
    # normalize temperature input to have a normalized scale for derivatives
    ut = unit_transform(temperature_domain...)
    iut = inv_unit_transform(temperature_domain...)
    unit_outer_temperature = ut.(outer_temperature)
    nout = length(outer_temperature)
    DT = zeros(nout, length(temperature_processes))
    UT = similar(DT) # uncertainty
    @threads for i in eachindex(temperature_processes) # for (i, G) in enumerate(temperature_processes)
        C = input_transformation(temperature_processes[i], iut) # inverse of temperature unit-scaling
        D = GaussianDistributions.derivative(C) # take the derivative w.r.t. T
        DT[:, i] = mean(D).(unit_outer_temperature) # record mean and var of derivative of each process
        UT[:, i] = var(D).(unit_outer_temperature)
    end
    # euclidean norm of temperature gradients of optical coefficients
    d, u = zeros(nout), zeros(nout)
    for i in 1:nout
        di, ui = @views DT[i, :], UT[i, :]
        d[i] = norm(di) # and first-order uncertainty propagation through "norm",
        u[i] = dot(ForwardDiff.gradient(norm, di).^2, ui)
    end
    # @. u = sqrt(u) # convert variance to std?
    return d, u # gradient values and their uncertainties
end
