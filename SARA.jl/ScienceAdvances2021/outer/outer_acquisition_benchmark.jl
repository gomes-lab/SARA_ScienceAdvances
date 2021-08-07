# benchmarks inner loop acquisition functions
using LinearAlgebra
using GaussianDistributions
using SARA
using SARA: random_sampling, uncertainty_sampling, ucb_sampling,
            ucb_acquisition,
            stripe_uncertainty_sampling, stripe_ucb_sampling,
            TemperatureProfile, stripe_to_global, get_relevant_T
using ForwardDiff
using Base.Threads: @threads
using SARA: oSARA, likelihood_ratio

############### generate sampling paths with different policies
function sampling_path(policy, G::Gaussian, x::AbstractMatrix, y::AbstractMatrix,
                       σ²::AbstractMatrix, stripe_conditions::AbstractVector, nsample::Int)
    C = G
    rmse, r2 = zeros(nsample), zeros(nsample) # rmse for data and estimated gradients
    indices = zeros(Int, nsample)
    helper(x, i) = @views vec(x[:, i]) # selects the columns indexed by i, and vectorizes the result
    for i in 1:nsample
        println("sample $i")
        xj = policy(C, stripe_conditions)
        j = findfirst(==(xj), stripe_conditions)
        indices[i] = j
        indices_i = @view indices[1:i]
        sort!(indices_i)
        xi, yi, σ²i = helper(x, indices_i), helper(y, indices_i), helper(σ², indices_i)
        C = conditional(G, xi, Gaussian(yi, σ²i), tol = 1e-6)
        μ = mean(C, vec(x))
        δ = vec(y) - μ
        rmse[i] = sqrt(mean(δ.^2 ./ vec(σ²))) # weighted root mean squared deviation
        r2[i] = likelihood_ratio(vec(y), μ, vec(σ²))
    end
    return rmse, r2
end

function outer_acquisition_benchmark(x::AbstractMatrix, y::AbstractMatrix,
                                σ²::AbstractMatrix, T_offset, stripe_conditions)
    subsampling = 2 # subsample observations per stripe (for quick testing only)
    stride = 2
    samplesperstripe = 16
    subsample(x::AbstractMatrix) = @view x[end+1-stride*samplesperstripe:stride:end, :]
    subsample(x::AbstractVector) = @view x[end+1-stride*samplesperstripe:stride:end]
    x = subsample(x)
    y = subsample(y)
    σ² = subsample(σ²)

    # standardize global data
    y = y .- mean(y) # otherwise r2 scores are initially negative
    y_normalization = std(y)
    y /= y_normalization
    σ² /= y_normalization^2 # noise variance

    tl = 30
    dl = .4
    k_scale = 1/4
    # kernel = k_scale^2 * oSARA([tl, dl]) # performance bug, sidestep with the following
    kernel = oSARA([tl, dl])
    σ² ./= k_scale^2

    # experimental parameters
    nstripes = 48 # how many stripes to acquire using active learning

    # stripe sampling acquisition functions
    constant_offset_bool = true
    nout = 32 # number of points on temperature curve that are observed per stripe
    constant_offset = Val(constant_offset_bool)
    # T_offset = passed as input # number of degrees from T_max we are generating data for
    T_range = range(T_offset[1], T_offset[2], length = nout)
    T_range = subsample(T_range) # [end-stride*samplesperstripe+1 : stride : end]
    T_offset = [maximum(T_range), minimum(T_range)]
    c_min, c_max = T_offset
    nout = length(T_range)
    relevant_T = get_relevant_T(constant_offset, c_min, c_max, nout)

    # random acquisition
    nrandom = 2
    random_funs = [random_sampling for _ in 1:nrandom]

    # stripe uncertainty sampling
    stripe_uncertainty = stripe_uncertainty_sampling(relevant_T)
    # stripe upper confidence bound sampling
    α_range = [1., 5.] # 10., 25., 50.] # coefficient of standard deviation in acquisition function
    ucb_funs = [ucb_sampling(α) for α in α_range]
    stripe_ucb_funs = [stripe_ucb_sampling(relevant_T, α) for α in α_range]

    policies = vcat(random_funs,
                    uncertainty_sampling, stripe_uncertainty,
                    ucb_funs, stripe_ucb_funs)

    npolicies = length(policies)
    rmse = zeros(nstripes, npolicies)
    r2 = zeros(nstripes, npolicies)
    G = Gaussian(kernel)
    @threads for i in eachindex(policies)
        println("policy $i of $(length(policies))")
        policy = policies[i]
        rmse[:, i], r2[:, i] = sampling_path(policy, G, x, y, σ², stripe_conditions, nstripes)
    end

    rmse_random_mean = vec(mean(rmse[:, 1:nrandom], dims = 2)) # averaging
    rmse_random_std = vec(std(rmse[:, 1:nrandom], dims = 2))
    rmse = hcat(rmse_random_mean, rmse[:, nrandom+1:end])

    r2_random_mean = vec(mean(r2[:, 1:nrandom], dims = 2))
    r2_random_std = vec(std(r2[:, 1:nrandom], dims = 2))
    r2 = hcat(r2_random_mean, r2[:, nrandom+1:end])

    if true
        f = h5open("outer_acquisition_benchmark.h5", "w")
        f["rmse"] = rmse
        f["r2"] = r2
        f["nstripes"] = nstripes
        f["kernel"] = ["oSARA"]
        f["temperature lengthscale"] = tl
        f["dwelltime lengthscale"] = dl
        f["npolicies"] = npolicies
        f["nrandom"] = nrandom
        f["ucb alpha range"] = α_range
        f["y_normalization"] = y_normalization
        f["metrics"] = ["rmse", "r2"]
        f["stride"] = stride
        f["samples per stripe"] = samplesperstripe
        f["subsampling"] = subsampling
        f["policy names"] = ["random", "uncertainty", "stripe uncertainty", "ucb", "stripe ucb"]
        close(f)
    end
    return rmse, r2
end

# load outer loop data
using HDF5
path = "SARA/ScienceAdvances2021/outer/data/" # put path to outer gradient data
# path = "ScienceAdvances2021/outer/"
# file = "Bi2O3_19F44_01_outer_loop_data.h5" # without input noise correction
file = "Bi2O3_19F44_01_outer_optical_gradients_l_2_input_noise_50.h5" # with input noise correction

f = h5open(path * file, "r")
dwelltimes = read(f, "dwelltimes")
temperatures = read(f, "temperatures")
gradients = read(f, "gradients")
var_gradients = read(f, "var(gradients)")
T_offset = read(f, "T_offset")
close(f)

# normalize global data
y_normalization = std(gradients)
gradients ./= y_normalization
var_gradients ./= y_normalization^2 # noise variance

x = tuple.(temperatures, dwelltimes)
y = gradients
σ² = var_gradients

# since the last condition of each column in the x array is offset from
# T_max by T_offset[2], need to add T_offset[2] back in to get the
# stripe condition
temperature_correction(x::NTuple{2}) = (x[1] + T_offset[2], x[2])
stripe_conditions = temperature_correction.(x[end, :])

rmse, r2 = outer_acquisition_benchmark(x, y, σ², T_offset, stripe_conditions)

r2_random = r2[:, 1]
r2_sigma = r2[:, 2]
r2_stripe_sigma = r2[:, 3]
nucb = (size(r2, 2)-3) ÷ 2
r2_ucb = r2[:, 4:3+nucb]
r2_stripe_ucb = r2[:, 4+nucb:end]

using Plots
plotly()
# names = ["random", "sigma", "ucb", "stripe sigma", "stripe ucb"]
plot(legend = :bottomright, xlabel = "number of stripes", ylabel = "R2")
plot!(r2_random, label = "random")
plot!(r2_sigma, label = "sigma")
plot!(r2_stripe_sigma, label = "stripe sigma")
plot!(r2_ucb, label = "ucb")
plot!(r2_stripe_ucb, label = "stripe ucb")
gui()

using HDF5
using Statistics
f = h5open("SARA/ScienceAdvances2021/outer/results/outer_acquisition_benchmark.h5")
r2 = read(f, "r2")
nrandom = read(f, "nrandom")
random = r2[:, 1:nrandom]

m = vec(mean(random, dims = 2))
s = vec(std(random, dims = 2))
threshold(p) = findfirst(>(p), m)
println(threshold(.7))
close(f)

f = h5open("SARA/ScienceAdvances2021/outer/results/outer_acquisition_benchmark_input_noise_0.h5")
r2 = read(f, "r2")
nrandom = read(f, "nrandom")
random = r2[:, 1:nrandom]

m = vec(mean(random, dims = 2))
s = vec(std(random, dims = 2))
threshold(p) = findfirst(>(p), m)
println(threshold(.7))
close(f)
