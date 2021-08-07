# converts exhaustive optical data to optical gradients
using LinearAlgebra
using GaussianDistributions
using SARA
using SARA: TemperatureProfile, stripe_to_global, get_relevant_T, iSARA
using Base.Threads: @threads
using SARA
using JSON
using HDF5

# path = "SARA/ScienceAdvances2021/inner/"
path = "../inner/"
include(path * "inner_load_data.jl")
using SARA: estimate_noise

path = "SARA/ScienceAdvances2021/inner/"
file = "Bi2O3_19F44_01_inner_loop_data.json"
f = JSON.parsefile(path * "data/" * file)
position, optical, rescaling_parameters, T_max, log10_τ = load_data(f)

# omat = [optical[k][j][i] for i in eachindex(optical[1][1]), j in eachindex(optical[1]), k in eachindex(optical)]
# println(maximum(omat))
########### estimating noise variance of normalized coefficient
l = .2 # from inner loop benchmarks
i = 3
x = position[i]
y = optical[i][1]
θ = rescaling_parameters[i]
k = iSARA(l, θ)
σ = estimate_noise(x, y, k) # make dependent on iSARA kernel
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
# dx = 100/1000 # uncertainty in position in mm
P = TemperatureProfile(dT, dx)
nout = 32 # we can still change this after the creation of the data
# constant_offset = Val(false)
# T_proportions = (.75, .99) # as a proportion of T_max, what data to convert to gradients
constant_offset_bool = true
constant_offset = Val(constant_offset_bool)
T_offset = [200, 10] # number of degrees from T_max we are generating data for
relevant_T = get_relevant_T(constant_offset, T_offset..., nout)

input_noise_bool = false
input_noise = Val(input_noise_bool)
nstripes = length(optical)
gradients = zeros(nout, nstripes)
var_gradients = zeros(nout, nstripes)
@threads for i in 1:nstripes
    println(i)
    kernel = iSARA(l, rescaling_parameters[i])
    gradients[:, i], var_gradients[:, i] = stripe_to_global(position[i], optical[i],
                                            σ, kernel,
                                            P, T_max[i], log10_τ[i], relevant_T, input_noise)
end

temperatures = zeros(nout, nstripes)
dwelltimes = zeros(nout, nstripes)
for i in 1:nstripes
    dwelltimes[:, i] .= log10_τ[i]
    temperatures[:, i] = relevant_T(T_max[i], log10_τ[i])
end

# savefile = "Bi2O3_19F44_01_outer_loop_data.h5"
# path = "../outer/data/"
# path = "SARA/ScienceAdvances2021/outer/data/"
path = ""
# how to read file name: l_2 means kernel lengthscale was .2,
# input_noise means input noise propagation was switched on
savefile = "Bi2O3_19F44_01_outer_optical_gradients_l_2_input_noise_0.h5"
f = h5open(path * savefile, "w")
f["dwelltimes"] = dwelltimes
f["temperatures"] = temperatures
f["gradients"] = gradients
f["var(gradients)"] = var_gradients
f["input noise"] = input_noise_bool
f["nstripes"] = nstripes
f["constant offset"] = constant_offset_bool
f["T_offset"] = T_offset
f["nout"] = nout
f["l"] = l
f["optical noise variance"] = σ^2
f["temperature uncertainty parameters"] = [dT, dx]
close(f)

# creates gradient map
# k = oSARA()
# G = Gaussian(k)
# temperatures = vec(temperatures)
# dwelltimes = vec(dwelltimes)
# x = tuple.(temperatures, dwelltimes)
# y = vec(target)
# σ² = vec(uncertainty) .+ .1
# C = G | (x, y, Diagonal(σ²))

# using Plots
# pyplot()
# T = 200:20.:1400
# τ = 2:.2:4
# surface(T, τ, (x,y)->mean(C)((x,y)))
# gui()
