### load data
using HDF5
# path = "SARA/NatCom2020/outer/results/"
savepath = "SARA/NatCom2020/outer/data/"
savefile = "Bi2O3_19F44_01_outer_optical_gradients_l_2_input_noise_50.h5"

f = h5open(savepath * savefile, "r")
temperatures = read(f, "temperatures")
dwelltimes = read(f, "dwelltimes")
gradients = read(f, "gradients")
var_gradients = read(f, "var(gradients)")
input_noise_bool = read(f, "input noise")
println(read(f, "temperature uncertainty parameters"))
l = read(f, "l")
println(l)
close(f)

# pre-process
using LinearAlgebra
using GaussianDistributions

x = tuple.(temperatures, dwelltimes)
y = gradients
σ² = var_gradients

y_normalization = std(y)
y ./= y_normalization
σ² ./= y_normalization^2

# subsample data
subsampling = 1
stride = 2 # small strides important to not get "waves"
samplesperstripe = 16
subsample(x, k::Int) = @view x[end+1-stride*k:stride:end, 1:subsampling:end]
# subsample(x, k::Int) = @view x[1:stride:stride*k, 1:subsampling:end]
# offset = 8
# subsample(x, k) = @view x[offset .+ (1:stride:stride*k), 1:subsampling:end]
x = subsample(x, samplesperstripe)
y = subsample(y, samplesperstripe)
σ² = subsample(σ², samplesperstripe)
x, y, σ² = vec(x), vec(y), vec(σ²)

# run regression
using SARA: oSARA, likelihood_ratio
k_scale = 1/4
k = oSARA([30, .4])
G = Gaussian(k)
@time C = conditional(G, x, Gaussian(y, Diagonal(σ²)), tol = 1e-6)

@time μ = mean(C).(x)
lr = likelihood_ratio(vec(y), vec(μ), vec(σ²))
println("likelihood ratio: $lr")

using Plots
# pyplot()
plotly()
n = 512
T = range(200, 1200, length = n + 1)
τ = range(2.5, 4, length = n)
g = mean(C).(tuple.(T, τ'))
@time heatmap(τ, T, g, title = savefile)
gui()

# for xrd
# savefile = "Bi2O3_19F44_01_outer_xrd_gradients_l_1_input_noise_50.h5"
# savefile = "Bi2O3_19F44_01_outer_xrd_gradients_l_1_input_noise_10.h5"
# savefile = "Bi2O3_19F44_01_outer_xrd_gradients_l_2_input_noise_10.h5"
# savefile = "Bi2O3_19F44_01_outer_xrd_gradients_large_input_noise.h5"
# k_scale = 1/2
# k = k_scale^2 * oSARA([30, .5])
