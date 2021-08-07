# benchmarks inner loop acquisition functions
using LinearAlgebra
using GaussianDistributions
using SARA
using SARA: stripe_uncertainty_sampling, TemperatureProfile, stripe_to_global, get_relevant_T
using ForwardDiff
using Base.Threads: @threads
using SARA: oSARA

# TODO: throw out sampled data!
############### generate sampling paths with different policies
function learning_path(policy, G::Gaussian, x::AbstractMatrix, y::AbstractMatrix,
                       σ²::AbstractMatrix, stripe_conditions, nsample::Int)
    n = 512
    T = range(200, 1200, length = n + 1) # plotting ranges
    τ = range(2.5, 4, length = n)
    gradients = zeros(n+1, n, nsample)
    C = G
    indices = zeros(Int, nsample)
    chosen_conditions = zeros(2, nsample)
    helper(x, i) = @views vec(x[:, i]) # selects the columns indexed by i, and vectorizes the result
    for i in 1:nsample
        println("stripe $i")
        xj = policy(C, stripe_conditions)
        chosen_conditions[:, i] .= xj
        j = findfirst(==(xj), stripe_conditions)
        indices[i] = j
        indices_i = @view indices[1:i]
        sort!(indices_i) # technically not necessary
        xi, yi, σ²i = helper(x, indices_i), helper(y, indices_i), helper(σ², indices_i)
        C = conditional(G, xi, Gaussian(yi, σ²i), tol = 1e-6)
        μ = mean(C, vec(x))
        δ = vec(y) - μ
        gradients[:, :, i] .= mean(C).(tuple.(T, τ'))
    end
    return T, τ, gradients, chosen_conditions
end

function outer_optical_learning(x::AbstractMatrix, y::AbstractMatrix, σ²::AbstractMatrix,
                                T_offset, stripe_conditions)
    # subsample observations per stripe
    stride = 2 # lower stride important to not get "waves"
    samplesperstripe = 16
    subsample(x::AbstractMatrix) = @view x[end+1-stride*samplesperstripe:stride:end, :]
    subsample(x::AbstractVector) = @view x[end+1-stride*samplesperstripe:stride:end]
    x = subsample(x)
    y = subsample(y)
    σ² = subsample(σ²)

    # standardize global data
    y_normalization = std(y)
    y /= y_normalization
    σ² /= y_normalization^2 # noise variance

    # experimental parameters
    nsample = 32 # number of active learning samples

    # set up relevant T function
    constant_offset_bool = true
    nout = 32
    constant_offset = Val(constant_offset_bool)
    T_range = range(T_offset[1], T_offset[2], length = nout)
    T_range = subsample(T_range)
    T_offset = [maximum(T_range), minimum(T_range)]
    c_min, c_max = T_offset
    nout = length(T_range)
    relevant_T = get_relevant_T(constant_offset, c_min, c_max, nout)

    # stripe uncertainty policy
    policy = stripe_uncertainty_sampling(relevant_T)

    T_length = 30.
    τ_length = .4
    k_scale = 1/4
    σ² ./= k_scale^2
    k = oSARA([T_length, τ_length])
    G = Gaussian(k)
    temperatures, dwelltimes, gradients, chosen_conditions =
                learning_path(policy, G, x, y, σ², stripe_conditions, nsample)
    if false
        f = h5open("outer_optical_learning.h5", "w")
        f["gradients"] = gradients
        f["nsample"] = nsample
        f["kernel"] = ["oSARA"]
        f["temperatures"] = collect(temperatures)
        f["dwelltimes"] = collect(dwelltimes)
        f["temperature lengthscale"] = T_length
        f["dwelltime lengthscale"] = τ_length
        f["y_normalization"] = y_normalization
        f["conditions"] = conditions
        f["dimensions"] = ["temperature", "dwelltime", "nstripes"]
        f["stride"] = stride
        f["samples per stripe"] = samplesperstripe
        close(f)
    end
    return temperatures, dwelltimes, gradients, chosen_conditions, stripe_conditions
end

# load outer loop data
using HDF5
path = "SARA/NatCom2020/outer/data/"
# path = "NatCom2020/outer/"
# file = "Bi2O3_19F44_01_outer_loop_data.h5" # without input noise
# file = "Bi2O3_19F44_01_outer_loop_input_noise_data.h5"
file = "Bi2O3_19F44_01_outer_optical_gradients.h5"
f = h5open(path * file, "r")
dwelltimes = read(f, "dwelltimes")
temperatures = read(f, "temperatures")
gradients = read(f, "gradients")
var_gradients = read(f, "var(gradients)")
T_offset = read(f, "T_offset")
close(f)

# IDEA: throw out data below τ = 2.5?
# ind = dwelltimes .≥ 2.5
# dwelltimes = dwelltimes[ind]
# temperatures = temperatures[ind]

# normalize global data
y_normalization = std(gradients)
gradients ./= y_normalization
var_gradients ./= y_normalization^2 # noise variance

x = tuple.(temperatures, dwelltimes)
y = gradients
σ² = var_gradients

temperature_correction(x::NTuple{2}) = (x[1] + T_offset[2], x[2])
stripe_conditions = temperature_correction.(x[end, :])

temperatures, dwelltimes, gradient_evolution, chosen_conditions = outer_optical_learning(x, y, σ², T_offset, stripe_conditions)

# using Plots
# plotly()
# heatmap(temperatures, dwelltimes, gradient_evolution[:, :, end]')
# scatter!(conditions[1, :], conditions[2, :])
