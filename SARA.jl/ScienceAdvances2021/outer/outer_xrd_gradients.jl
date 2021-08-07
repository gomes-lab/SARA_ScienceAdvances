# converts exhaustive xrd data to xrd gradients
using LinearAlgebra
using GaussianDistributions
using SARA
using SARA: TemperatureProfile, stripe_to_global, get_relevant_T, iSARA
using Base.Threads: @threads
using SARA: iSARA, oSARA
using JSON
using HDF5

path = "../inner/"
# path = "SARA/NatCom2020/inner/"
include(path * "inner_estimate_noise.jl")
include(path * "inner_load_data.jl")

datapath = "SARA/NatCom2020/outer/data/"
# datapath = ".."
file = "Bi2O3_19F44_01_outer_xrd_data.h5"
f = h5open(datapath * file)
positions = read(f, "positions")
temperatures = read(f, "temperatures")
dwelltimes = read(f, "dwelltimes")
spectrograms = read(f, "spectrograms")
close(f)
conditions = tuple.(temperatures, dwelltimes)

########### match rescaling parameters from optical coefficients to xrd data
# returns true if dwell time and peak temperature are the same
same_conditions(f) = g->same_conditions(f, g)
same_conditions(f, g) = all(f .== g)

innerpath = "SARA/NatCom2020/inner/"
# path = ".."
file = "Bi2O3_19F44_01_inner_loop_data.json"
f = JSON.parsefile(innerpath * "data/" * file)
_, _, optical_rescaling, optical_temperatures, optical_dwelltimes = load_data(f)
optical_conditions = tuple.(optical_temperatures, optical_dwelltimes)

rescaling_parameters = []
for i in eachindex(conditions)
    j = findfirst(same_conditions(conditions[i]), optical_conditions)
    if !isnothing(j)
        push!(rescaling_parameters, optical_rescaling[j])
    end
end

# 2. dimensionality reduction
using SARA: legendre_coefficients, svd_coefficients
ncoeff = 16
coefficients = svd_coefficients(spectrograms, ncoeff)
# coefficients = legendre_coefficients(spectrograms, ncoeff)
coefficients ./= maximum(coefficients)

########### estimating noise variance of normalized coefficient
i = 3
x = positions[:, i]
y = coefficients[1, :, i]
using CovarianceFunctions
const Kernel = CovarianceFunctions

l = .1 # from inner loop benchmarks
θ = rescaling_parameters[i]
k = iSARA(l, θ)
σ = estimate_noise(x, y, k)
println(σ)

# using Plots
# plotly()
# C = Gaussian(k) | (x, y, σ^2)
# plot(x, mean(C, x), ribbon = 2std(C).(x))
# scatter!(x, y)

############## convert to global data
# dT = 20 # uncertainty in peak temperature in C
# dx = 10/1000 # uncertainty in position in mm
dT = 25 # uncertainty in peak temperature in C
dx = 50/1000 # uncertainty in position in mm
P = TemperatureProfile(dT, dx)
nout = 32 # we can still change this after the creation of the data
# constant_offset = Val(false)
# T_proportions = (.75, .99) # as a proportion of T_max, what data to convert to gradients
constant_offset_bool = true
constant_offset = Val(constant_offset_bool)
T_offset = [200, 10] # number of degrees from T_max we are generating data for
relevant_T = get_relevant_T(constant_offset, T_offset..., nout)

input_noise_bool = true
input_noise = Val(input_noise_bool)
nstripes = size(coefficients, 3)
gradients = zeros(nout, nstripes)
var_gradients = zeros(nout, nstripes)
@threads for i in 1:nstripes
    println(i)
    pos = @view positions[:, i]
    coeff = [@view coefficients[j, :, i] for j in 1:size(coefficients, 1)]
    kernel = iSARA(l, rescaling_parameters[i])
    gradients[:, i], var_gradients[:, i] = stripe_to_global(pos, coeff,
                                            σ, kernel,
                                            P, temperatures[i], dwelltimes[i],
                                            relevant_T, input_noise)
end

outer_temperatures = zeros(nout, nstripes)
outer_dwelltimes = zeros(nout, nstripes)
for i in 1:nstripes
    outer_dwelltimes[:, i] .= dwelltimes[i]
    outer_temperatures[:, i] = relevant_T(temperatures[i], dwelltimes[i])
end

savepath = "SARA/NatCom2020/outer/data/"
# savefile = "Bi2O3_19F44_01_outer_xrd_gradients_large_input_noise.h5"
savefile = "Bi2O3_19F44_01_outer_xrd_gradients_l_1_input_noise_50_new_profile.h5"
# savefile = "Bi2O3_19F44_01_outer_xrd_gradients_no_input_noise.h5"
f = h5open(savepath * savefile, "w")
f["dwelltimes"] = outer_dwelltimes
f["temperatures"] = outer_temperatures
f["gradients"] = gradients
f["var(gradients)"] = var_gradients
f["input noise"] = input_noise_bool
f["nstripes"] = nstripes
f["constant offset"] = constant_offset_bool
f["T_offset"] = T_offset
f["coefficients"] = coefficients
f["coefficient type"] = "svd" # "legendre"
f["nout"] = nout
f["l"] = l
f["coefficient noise variance"] = σ^2
f["temperature uncertainty parameters"] = [dT, dx]
close(f)

# using GaussianDistributions: optimize
# k = (x, y) -> Kernel.NN(100.)(x./l, y./l)
# kernel = Kernel.Lengthscale(Kernel.MaternP(2), l)
# k = (θ) -> ((x, y) -> exp(θ[1]) * Kernel.NN(exp(θ[2]))(x./exp(θ[3]), y./exp(θ[3])))
# θ = zeros(3)
# println(θ)
# θ = optimize(k, θ, x, y, σ^2)
# println(θ)
# k = k(θ)
