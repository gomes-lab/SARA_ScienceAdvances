using LinearAlgebra
using GaussianDistributions
using SARA
using SARA: random_sampling, uncertainty_sampling, outer_sampling,
        TemperatureProfile, stripe_to_global
using ForwardDiff
using Base.Threads
using SARA: oSARA

function optical_gradient_map(x::AbstractMatrix, y::AbstractMatrix, σ²::AbstractMatrix)
    subsampling = 1
    stride = 1
    subsample(x, k::Int) = @view x[end+1-stride*k:stride:end, 1:subsampling:end]
    T_length = [30] # range(20, 35, step = 5) # different T length scales
    nsamples = [16] # vcat(1, 2, 4, 6, 8, 12, 16)
    k_scale = 1/4 # kernel scaling hyper-parameter
    σ² ./= k_scale^2
    τ_length = [.4] # τ length scales
    n = 512
    T = range(200, 1200, length = n + 1) # plotting ranges
    τ = range(2.5, 4, length = n)
    gradients = zeros(length(T), length(τ), length(nsamples), length(T_length), length(τ_length))
    for i in eachindex(T_length)
        T_l = T_length[i]
        for (j, τ_l) in enumerate(τ_length)
            @threads for ik in eachindex(nsamples) # number of gradient samples per stripe
                println(i, j, ik)
                k = nsamples[ik]
                xk = subsample(x, k)
                yk = subsample(y, k)
                σ²k = subsample(σ², k)

                # kernel = k_scale^2 * oSARA([T_l, τ_l])
                kernel = oSARA([T_l, τ_l])
                G = Gaussian(kernel)
                xk = vec(xk)
                yk = vec(yk)
                # μ_yk = mean(yk)
                # yk = yk .- μ_yk
                σ²k = vec(σ²k)
                @time C = conditional(G, xk, Gaussian(yk, Diagonal(σ²k)), tol = 1e-6)
                # C = C + μ_yk
                gij = @view gradients[:, :, ik, i, j]
                # gij = vec(gij)
                # @time gij .= mean(C, tuple.(T, τ'))
                @time gij .= mean(C).(tuple.(T, τ'))
            end
        end
    end
    if true
        f = h5open("outer_optical_gradient_map_stride_1.h5", "w")
        f["temperatures"] = collect(T)
        f["dwelltimes"] = collect(τ)
        f["gradients"] = gradients
        f["nsample"] = collect(nsamples)
        f["temperature lengthscales"] = collect(T_length)
        f["dwelltime lengthscales"] = collect(τ_length)
        f["kernel scale"] = k_scale
        f["dimensions"] = ["temperature", "dwelltime", "nsample",
                            "temperature lengthscale", "dwelltime lengthscale"]
        f["subsampling"] = subsampling
        f["stride"] = stride
        close(f)
    end
    return T, τ, gradients
end

# load outer loop data
using HDF5
# optical data
# path = "SARA/NatCom2020/outer/data/"
# path = "NatCom2020/"
# file = "Bi2O3_19F44_01_outer_loop_data.h5" # without input noise
# file = "Bi2O3_19F44_01_outer_loop_input_noise_data.h5"
# file = "Bi2O3_19F44_01_outer_loop_input_noise_constant_offset.h5"
# xrd data
path = "SARA/NatCom2020/outer/data/"
# path = "NatCom2020/"
# file = "Bi2O3_19F44_01_outer_xrd_gradients_input_noise.h5"
file = "Bi2O3_19F44_01_outer_optical_gradients_l_2_input_noise_50.h5"

f = h5open(path * file, "r")
temperatures = read(f, "temperatures")
dwelltimes = read(f, "dwelltimes")
gradients = read(f, "gradients")
var_gradients = read(f, "var(gradients)")
close(f)

# normalize global data
y_normalization = std(gradients)
gradients ./= y_normalization
var_gradients ./= y_normalization^2 # noise variance

x = tuple.(temperatures, dwelltimes)
y = gradients
σ² = var_gradients

T, τ, gradients = optical_gradient_map(x, y, σ²)

################################################################################
# plot it
# using HDF5
# # f = h5open("SARA/NatCom2020/results/outer_optical_gradient_map.h5")
# f = h5open("outer_optical_gradient_map.h5")
#
# T = read(f, "temperatures")
# τ = read(f, "dwelltimes")
# gradients = read(f, "gradients")
# A = gradients[:, :, 1, 1, 1]
#
# using Plots
# pyplot()
# heatmap(T, τ, A')
# gui()
# close(f)

################################################################################

# calculate single map
# subsample(x, k::Int) = @view x[end+1-k:end, :]
# n = 8
# x = subsample(x, n)
# y = subsample(y, n)
# σ² = subsample(σ², n)

# using Kernel
# # from marginal likelihood opt:
# T_l = 26.629076190570203
# τ_l = .3 # 15.924969663921164
# NN_const = 1111.2566343353083
# # T_l, τ_l = 10., .2
# # kernel = oSARA([T_l, τ_l])
# kernel = Kernel.NN(NN_const)
# scaling = [T_l, τ_l]
# scale(k) = (x, y) -> k(x./scaling, y./scaling)
# kernel = scale(kernel)
# G = Gaussian(kernel)
# x = vec(x)
# y = vec(y)
# # μ_yk = mean(yk)
# # yk = yk .- μ_yk
# σ² = vec(σ²)
# @time C = conditional(G, x, Gaussian(y, Diagonal(σ²)), tol = 1e-6)
# # C = C + μ_yk
# n = 64
# T = range(200, 1200, length = n) # plotting ranges
# τ = range(2.5, 4, length = n + 1)
# gradients = mean(C).(tuple.(T, τ'))
# heatmap(T, τ, gradients')

# using Plots
# pyplot()
# n = 64
# T = range(200, 1200, length = n)
# τ = range(2.5, 4, length = n)
# heatmap(T, τ, (x, y)->mean(C)((x, y)))
# gui()

# marginal likelihood optimization of nn kernel
# function nn_kernel(θ)
#     l1, l2, σ = θ
#     scaling = [l1, l2]
#     scale(k) = (x, y) -> k(x./scaling, y./scaling)
#     return scale(Kernel.NN(σ))
# end
# using GaussianDistributions: optimize
# θ = [40, 0.3, 1000]
#
# subsample_spec(x, k::Int) = @view x[:, 1:k]
# d = 8
# xk, yk, σ²k = subsample_spec(x, d), subsample_spec(y, d), subsample_spec(σ², d)
# xk, yk, σ²k = vec.((xk, yk, σ²k))
# optimize(nn_kernel, θ, xk, Gaussian(yk, σ²k))
