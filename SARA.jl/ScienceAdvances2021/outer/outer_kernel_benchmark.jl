# comparing EQ, MaternP, NN kernel with energetic norm for outer loop data
using LinearAlgebra
using SARA
using SARA: iSARA, random_sampling
using CovarianceFunctions
const Kernel = CovarianceFunctions
using GaussianDistributions
using Base.Threads

# generalized coefficient of determination, where errors are weighted by their standard deviation
# McFadden (in the context of logistic regression), (log?) likelihood ratio index
# x is data, y is model prediction, σ is expected noise for each data point
function likelihood_ratio(x::AbstractVector, y::AbstractVector, σ::AbstractVector = ones(length(x)))
    δ = (x .- mean(x)) ./ σ
    tss = sum(abs2, δ)
    δ = (x .- y) ./ σ
    ess = sum(abs2, δ)
    r2 = 1 - ess / tss
end

################# setup kernels for comparison ##############
# θ are stripe-dependent rescaling parameters
# LSA-bias is encoded in last parameter tuple
# l is length scale for all kernels
get_kernels(l) = get_kernels(l...)
function get_kernels(T_length::Real, τ_length::Real)
    # list of base kernels
    k1 = Kernel.EQ()
    k2 = Kernel.MaternP(2)
    k3 = Kernel.NN(1000.) # parameter governs size of bias coefficients in neural network, put proportional to temperature domain
    base_kernels = [k1, k2, k3]
    # equip base kernels with energetic norm
    # putlength(k) = Kernel.Lengthscale(k, l)
    # A = Diagonal(inv.([T_length, τ_length]).^2)
    # putenergy(k) = Kernel.Energetic(k, A)
    # kernels = putenergy.(base_kernels)
    scaling = [T_length, τ_length]
    scale(k) = (x, y) -> k(x./scaling, y./scaling)
    kernels = scale.(base_kernels)
    return kernels
end

############### generate sampling paths with different kernels
function sampling_path(G::Gaussian, x::AbstractMatrix,
                       y::AbstractMatrix,  σ²::AbstractMatrix, samplesizes)
    C = G
    nsample = length(samplesizes)
    rmse, r2 = zeros(nsample), zeros(nsample)
    indices = zeros(Int, 0)
    n, m = size(x)
    helper(x, i) = @views vec(x[:, i]) # selects the columns indexed by i, and vectorizes the result
    for (i, s) in enumerate(samplesizes) # random sampling
        indices = rand(1:m, s) # choose s random columns
        sort!(indices)
        xi, yi, σ²i = helper(x, indices), helper(y, indices), helper(σ², indices)
        C = conditional(G, xi, Gaussian(yi, σ²i), tol = 1e-6)
        μ = mean(C, vec(x))
        δ = vec(y) - μ
        rmse[i] = sqrt(mean(δ.^2 ./ vec(σ²))) # weighted root mean squared deviation
        r2[i] = likelihood_ratio(vec(y), μ, vec(σ²))
    end
    return rmse, r2
end

# using random sampling
function outer_kernel_benchmark!(rmse, r2, kernels, x, y, σ², samplesizes)
    nexp = size(rmse, 1)
    for (j, k) in enumerate(kernels)
        println("kernel $j of $(length(kernels))")
        G = Gaussian(k)
        @threads for i in 1:nexp
            d, r = sampling_path(G, x, y, σ², samplesizes)
            rmse[i, :, j] = d
            r2[i, :, j] = r
        end
    end
    return rmse
end

function energetic_kernel_benchmark(x, y, σ², nexp, nkernel, samplesizes, samplesperstripe)
    nsample = length(samplesizes) # number of different sample sizes [1, ..., nsample]
    # range of plausible length scales
    T_length = range(20, 40, step = 10)
    τ_length = range(.2, .4, step = .1)
    nlength = length(T_length) * length(τ_length)
    rmse = zeros(nexp, nsample, nkernel, length(T_length), length(τ_length))
    r2 = zeros(nexp, nsample, nkernel, length(T_length), length(τ_length))
    # first, calculate with energetic input norm
    for (i, T_l) in enumerate(T_length)
        for (j, τ_l) in enumerate(τ_length)
            println("lengthscale $(length(τ_length)*(i-1) + j) of $nlength")
            kernels = get_kernels((T_l, τ_l))
            rmse_ij = @view rmse[:, :, :, i, j]
            r2_ij = @view r2[:, :, :, i, j]
            outer_kernel_benchmark!(rmse_ij, r2_ij, kernels, x, y, σ², samplesizes)
        end
    end

    if true
        f = h5open("outer_kernel_benchmark_energetic.h5", "w")
        f["weighted rmse"] = rmse
        f["likelihood ratio"] = r2
        f["temperature lengthscales"] = collect(T_length) # in resultss
        f["dwelltime lengthscales"] = collect(τ_length)
        f["nexp"] = nexp
        f["sample sizes"] = collect(samplesizes)
        f["nlength"] = nlength
        f["nkernel"] = nkernel
        f["kernel names"] = ["EQ", "Matern_2", "NN"]
        f["dimensions"] = ["nexp", "nsample", "nkernel", "temperature lengthscales", "dwelltime lengthscales"]
        f["samples per stripe"] = samplesperstripe
        close(f)
    end
    return rmse, r2
end

function uniform_kernel_benchmark(x, y, σ², nexp, nkernel, samplesizes, samplesperstripe)
    nsample = length(samplesizes) # number of different sample sizes [1, ..., nsample]
    # range of plausible length scales
    lengthscales = [.25, .5, 1., 2., 4., 8., 16., 32.]
    rmse = zeros(nexp, nsample, nkernel, length(lengthscales))
    r2 = zeros(nexp, nsample, nkernel, length(lengthscales))
    for (i, l) in enumerate(lengthscales)
        println("lengthscale $i of $(length(lengthscales))")
        kernels = get_kernels((l, l))
        rmse_i = @view rmse[:, :, :, i]
        r2_i = @view r2[:, :, :, i]
        outer_kernel_benchmark!(rmse_i, r2_i, kernels, x, y, σ², samplesizes)
    end

    if true
        f = h5open("outer_kernel_benchmark_uniform.h5", "w")
        f["weighted rmse"] = rmse
        f["likelihood ratio"] = r2
        f["lengthscales"] = collect(lengthscales) # in resultss
        f["nexp"] = nexp
        f["sample sizes"] = collect(samplesizes)
        f["nlength"] = length(lengthscales)
        f["nkernel"] = nkernel
        f["kernel names"] = ["EQ", "Matern_2", "NN"]
        f["dimensions"] = ["nexp", "nsample", "nkernel", "lengthscales"]
        f["samples per stripe"] = samplesperstripe
        close(f)
    end
    return rmse, r2
end


function outer_kernel_benchmark_driver(x::AbstractMatrix, y::AbstractMatrix, σ²::AbstractMatrix)
    # subsample observations per
    subsample(x, k::Int) = @view x[end+1-k:end, :]
    samplesperstripe = 8
    x = subsample(x, samplesperstripe)
    y = subsample(y, samplesperstripe)
    σ² = subsample(σ², samplesperstripe)

    # standardize global data
    y = y .- mean(y)
    y_normalization = std(y)
    y /= y_normalization
    σ² /= y_normalization^2 # noise variance

    # experimental parameters
    nexp = 1 # number of independent experiments per nsample
    nkernel = 3 # number of kernels

    samplemax = 64
    samplestep = 32 # number of sample
    samplesizes = range(32, samplemax, step = samplestep)
    energetic_kernel_benchmark(x, y, σ², nexp, nkernel, samplesizes, samplesperstripe)
    uniform_kernel_benchmark(x, y, σ², nexp, nkernel, samplesizes, samplesperstripe)
end

# load outer loop data
using HDF5
path = "/Users/sebastianament/Documents/SEA/XRD Analysis/SARA/Bi2O3_19F44_01/"
# path = "NatCom2020/"
# file = "Bi2O3_19F44_01_outer_loop_data.h5" # without input noise
file = "Bi2O3_19F44_01_outer_loop_input_noise_data.h5"
f = h5open(path * file, "r")
dwelltimes = read(f, "dwelltimes")
temperatures = read(f, "temperatures")
gradients = read(f, "gradients")
var_gradients = read(f, "var(gradients)")
close(f)

# pre-process data
x = tuple.(temperatures, dwelltimes)
y = gradients
σ² = var_gradients # uncertainty in y

rmse, r2 = outer_kernel_benchmark_driver(x, y, σ²)

# mrmse = reshape(mean(rmse, dims = 1), size(rmse)[2:4]..., :)
# mr2 = reshape(mean(r2, dims = 1), size(r2)[2:4]..., :)
