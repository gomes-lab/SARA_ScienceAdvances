# comparing iSARA policy to canonical acquisition functions
using Plots
using HDF5
using Statistics
function plot_kernel_benchmark(f)
    # error = read(f, "mean_squared_error")
    # error = read(f, "rmse")
    error = read(f, "r2")
    μ = mean(error, dims = (2, 5)) # average over independent experiments and stripes
    σ = std(error, dims = (2, 5))
    nsample, nexp, nlength, nkernel = read(f, "nsample"), read(f, "nexp"), read(f, "nlength"), read(f, "nkernel")
    μ = reshape(μ, nsample, nkernel, nlength)
    σ = reshape(σ, nsample, nkernel, nlength)
    # for each kernel, use length scale with best performance
    lind = zeros(Int, nkernel)
    for i in 1:nkernel
        A = @view μ[end, i, :]
        # lind[i] = argmin(A)
        lind[i] = argmax(A)
    end
    A = zeros(nsample, nkernel)
    B = zeros(nsample, nkernel)
    for i in 1:nkernel
        @. A[:, i] = μ[:, i, lind[i]]
        @. B[:, i] = σ[:, i, lind[i]]
    end
    n = prod(size.((error,), (2, 5)))
    confidence = maximum(B) / sqrt(n)
    println("confidence = $confidence")
    lengthscales = read(f, "lengthscales")
    println("lengthscales = $(lengthscales[lind])")
    names = read(f, "kernel names")
    plot(xlabel = "# of points", ylabel = "R2") #i, yscale = :log10)
    plot!(A, label = permutedims(names), linewidth = 2) #, linestyle = :auto)
    gui()
end

path = "SARA/NatCom2020/results/"
file = "inner_kernel_benchmark_short.h5"
# file = "inner_kernel_benchmark_short_endpoints.h5"
f = h5open(path * file)


plotly()
# plotlyjs()
# pyplot()
# Plots.scalefontsizes(1.5)
plot_kernel_benchmark(f)
# savefig("inner_kernel_plot.pdf")
