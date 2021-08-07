# comparing the iSARA kernel to canonical kernels
using SARA
using SARA: iSARA, random_sampling
using CovarianceFunctions
const Kernel = CovarianceFunctions
using GaussianDistributions

################# load data
include("inner_load_data.jl")
using SARA: estimate_noise

################# setup kernels for comparison ##############
# θ are stripe-dependent rescaling parameters
# LSA-bias is encoded in last parameter tuple
# l is length scale for all kernels
function get_kernels(l, θ)
    k1 = Kernel.Lengthscale(Kernel.EQ(), l)
    k2 = Kernel.Lengthscale(Kernel.MaternP(2), l)
    k3 = iSARA(l)

    # only LSA-inspired rescaling
    k4 = iSARA(l, θ[end])  # last parameter tuple

    # with all rescaling parameters
    k5 = iSARA(l, θ)
    kernels = [k1, k2, k3, k4, k5]
    return kernels
end

############### generate sampling paths with different kernels
function sampling_path(policy, G::Gaussian, x::AbstractVector, y::AbstractVector, σ::Real, nsample::Int)
    C = G
    rmse, r2 = zeros(nsample), zeros(nsample)
    indices = zeros(Int, 0)
    sstot = sum(abs2, y .- mean(y)) # total sum of squares
    for i in 1:nsample # random sampling
        xi = policy(C, x)
        append!(indices, findfirst(==(xi), x))
        sort!(indices)
        C = G | @views (x[indices], y[indices], σ^2)
        μ = mean(C, x)
        δ = y - μ
        rmse[i] = sqrt(mean(δ.^2)) # RMSD (root mean squared deviation)
        r2[i] = 1 - (sum(abs2, δ) / sstot) # coefficient of determination
    end
    return rmse, r2
end

# on a single stripe using random sampling
function kernel_benchmark!(rmse, r2, kernels, x, y, σ)
    nsample, nexp = size(rmse)[1:2]
    for (j, k) in enumerate(kernels)
        G = Gaussian(k)
        Base.Threads.@threads for i in 1:nexp
            d, r = sampling_path(random_sampling, G, x, y, σ, nsample)
            rmse[:, i, j] = d
            r2[:, i, j] = r
        end
    end
    return rmse
end

function kernel_benchmark_driver(position, optical, rescaling_parameters,
                                 σ::Real, y_normalization::Real)
    nstripes = length(position)
    nkernel = 5
    nsample = 64
    lengthscales = collect(range(.05, .25, step = .05))
    nlength = length(lengthscales)
    nexp = 128 # number of independent experiments per stripe
    rmse = zeros(nsample, nexp, nkernel, nlength, nstripes)
    r2 = zeros(nsample, nexp, nkernel, nlength, nstripes)
    for i in 1:nstripes
        println("stripe $i of $nstripes")
        θ = rescaling_parameters[i]
        for (j, l) in enumerate(lengthscales)
            kernels = get_kernels(l, θ)

            rmse_i = @view rmse[:, :, :, j, i]
            r2_i = @view r2[:, :, :, j, i]

            x = position[i]
            y = optical[i][1] # for simplicity, benchmark with first and most significant optical coefficient
            y .-= mean(y) # normalized and centered
            y /= y_normalization
            kernel_benchmark!(rmse_i, r2_i, kernels, x, y, σ)
        end
    end

    if true
        f = h5open("inner_kernel_benchmark.h5", "w")
        f["rmse"] = rmse
        f["r2"] = r2
        f["lengthscales"] = lengthscales
        f["nexp"] = nexp
        f["nsample"] = nsample
        f["nlength"] = nlength
        f["nstripes"] = nstripes
        f["nkernel"] = nkernel
        f["kernel names"] = ["EQ", "Matern_2", "iSARA", "iSARA+LSA", "iSARA+RGB"]
        close(f)
    end
    return rmse, r2
end

# path = "/Users/sebastianament/Documents/SEA/XRD Analysis/SARA/Bi2O3_19F44_01/"
path = ""
file = "Bi2O3_19F44_01_inner_loop_data.json"
f = JSON.parsefile(path * file)
position, optical, rescaling_parameters, T_max, log10_τ = load_data(f)

# estimating standard deviation of noise of normalized coefficient
x = position[1]
y = optical[1][1]
y .-= mean(y)
y_normalization = std(y)
# y_normalization = maximum(abs, y)
y /= y_normalization
σ = estimate_noise(x, y)
rmse, r2 = kernel_benchmark_driver(position, optical, rescaling_parameters, σ, y_normalization)

# mrmse = reshape(mean(rmse, dims = (2, 5)), 64, 5)
# mr2 = reshape(mean(r2, dims = (2, 5)), 64, 5)

# c = [x[1] for x in optical]
