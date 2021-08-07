# benchmarks inner loop acquisition functions
using LinearAlgebra
using GaussianDistributions
using SARA
using SARA: random_sampling, uncertainty_sampling, inner_sampling,
        integrated_uncertainty_sampling, integrated_uncertainty, iSARA
using ForwardDiff
using Base.Threads: @threads
using CovarianceFunctions
const Kernel = CovarianceFunctions

# computes the mean derivative of the Gaussian process conditioned on (x, y)
function get_gradient(G::Gaussian, x::AbstractVector, y::AbstractVector, σ::Real)
    C = G | (x, y, σ^2)
    D = GaussianDistributions.derivative(C)
    ∂y = mean(D, x)
end

############### generate sampling paths with different policies
function sampling_path(policy, G::Gaussian, x::AbstractVector, y::AbstractVector,
                       ∂y::AbstractVector, σ::Real, nsample::Int)
    C = G
    indices = zeros(Int, 0)
    rmse, r2 = zeros(nsample, 2), zeros(nsample, 2) # rmse for data and estimated gradients
    sstot = sum(abs2, y .- mean(y)) # total sum of squares
    ∂sstot = sum(abs2, ∂y .- mean(∂y))
    for i in 1:nsample # random sampling
        xi = policy(C, x)
        append!(indices, findfirst(==(xi), x))
        sort!(indices)
        C = G | @views (x[indices], y[indices], σ^2)
        # RMSE (root mean squared error) and R2
        μ = mean(C, x)
        δ = μ - y
        rmse[i, 1] = sqrt(mean(δ.^2))
        r2[i, 1] = 1 - (sum(abs2, δ) / sstot)
        # RMSE and R2 of gradient prediction
        D = GaussianDistributions.derivative(C) # derivative process
        ∂μ = mean(D, x)
        ∂δ = ∂μ - ∂y
        rmse[i, 2] = sqrt(mean(δ.^2))
        r2[i, 2] = 1 - (sum(abs2, ∂δ) / ∂sstot)
    end
    return rmse, r2
end

# on a single stripe
function acquisition_benchmark!(rmse, r2, policies, k, x, y, σ)
    nsample = size(rmse, 1)
    G = Gaussian(k)
    ∂y = get_gradient(G, x, y, σ) # computes gradient "ground truth" on all available data
    for (i, policy) in enumerate(policies)
        println("policy $i of $(length(policies))")
        @time rmse[:, :, i], r2[:, :, i] = sampling_path(policy, G, x, y, ∂y, σ, nsample)
    end
    return rmse, r2
end

function acquisition_benchmark_driver(position, optical, rescaling_parameters,
                                      σ::Real, y_normalization::Real)
    nstripes = length(position)
    # nrandom = 32 # uncomment these two lines to run multiple random samplings to report statistics
    # policies = [random_sampling for _ in 1:nrandom]
    policies = [random_sampling, uncertainty_sampling,
                integrated_uncertainty_sampling(σ), inner_sampling(σ)]
    npolicies = length(policies)
    nsample = 128 # number of samples the active learning algorithm collects
    nmetrics = 2 # number of error metrics to compute
    rmse = zeros(nsample, nmetrics, npolicies, nstripes)
    r2 = zeros(nsample, nmetrics, npolicies, nstripes)
    @threads for i in 1:nstripes
        println("stripe $i of $nstripes")
        θ = rescaling_parameters[i]
        # uncomment these two lines to run the acquisition benchmark with the specialized SARA kernel
        # kernel for inner loop acquisition (l cross-validated from kernel benchmarks)
        l = .2 # is best from kernel benchmarks
        k = iSARA(l, θ)
        # l = .05 # is best from kernel benchmarks for EQ
        # k = Kernel.Lengthscale(Kernel.EQ(), l)
        rmse_i = @view rmse[:, :, :, i]
        r2_i = @view r2[:, :, :, i]
        x = position[i]
        y = optical[i][1] # for simplicity, benchmark with first and most significant optical coefficient
        y = y .- mean(y) # normalized and centered
        y /= y_normalization
        acquisition_benchmark!(rmse_i, r2_i, policies, k, x, y, σ)
    end

    if true
        # f = h5open("inner_random_acquisition_benchmark.h5", "w")
        # f["policy names"] = ["random" for _ in 1:npolicies]
        f = h5open("inner_acquisition_benchmark.h5", "w")
        f["policy names"] = ["random", "uncertainty", "int. unc.", "int. grad. unc."]
        f["rmse"] = rmse
        f["r2"] = r2
        f["nsample"] = nsample
        # f["kernel"] = ["EQ"]
        f["kernel"] = ["iSARA+RGB"]
        f["nstripes"] = nstripes
        f["npolicies"] = npolicies
        f["y_normalization"] = y_normalization
        f["metrics"] = ["rmse", "r2", "rmse of gradient", "r2 of gradient"]
        close(f)
    end
    return rmse, r2
end

include("inner_load_data.jl")
using SARA: estimate_noise

path = "SARA/ScienceAdvances2021/inner/data/"
file = "Bi2O3_19F44_01_inner_loop_data.json"
f = JSON.parsefile(path * file)

position, optical, rescaling_parameters, T_max, log10_τ = load_data(f)

# estimating noise variance of normalized coefficient
x = position[1]
y = optical[1][1]
y = y .- mean(y)
y_normalization = std(y)
y /= y_normalization # std(y)
# θ = rescaling_parameters[1]
# k = iSARA(.2, θ)
l = .05
k = Kernel.Lengthscale(Kernel.EQ(), l)
σ = estimate_noise(x, y, k)
println(σ)
rmse, r2 = acquisition_benchmark_driver(position, optical, rescaling_parameters, σ, y_normalization)

# mrmse = reshape(mean(rmse, dims = (3, 4)), 64, 5)

# random analysis
m = mean(r2, dims = (3, 4))[:, 2]
s = std(r2, dims = (3, 4))[:, 2]

# standard errors
s ./= sqrt(prod(size(r2)[3:4]))
threshold(p) = findfirst(>(p), m)
println(threshold(.8))
println(threshold(.9))
println(s)
