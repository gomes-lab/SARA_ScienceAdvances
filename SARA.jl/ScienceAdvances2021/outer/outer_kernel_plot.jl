using HDF5

# f = h5open("SARA/NatCom2020/results/outer_kernel_benchmark_energetic.h5")
# f = h5open("SARA/NatCom2020/results/outer_kernel_benchmark_uniform.h5")
f = h5open("SARA/NatCom2020/results/outer_kernel_benchmark_energetic_no_input_noise_short.h5")


using Plots
pyplot()

r2 = read(f, "likelihood ratio")
nsamples = read(f, "sample sizes")

tl = read(f, "temperature lengthscales")
dl = read(f, "dwelltime lengthscales")
names = read(f, "kernel names")

using Statistics
mr2 = median(r2, dims = 1)
mr2 = reshape(mr2, size(r2)[2:end])

# choose best length scale
br2 = zeros(size(mr2, 1), size(mr2, 2))
for i in eachindex(names)
    j = argmax(@view mr2[end, i, :, :])
    println(j)
    println(tl[j[1]])
    println(dl[j[2]])
    br2[:, i] = mr2[:, i, j[1], j[2]]
end
plot!(xlabel = "number of samples", ylabel = "R^2_sigma")
plot!(nsamples, br2, label = permutedims(names), legend = :bottomright)
gui()
