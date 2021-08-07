using HDF5
using Statistics
function plot_acquisition_benchmark(f)
    # error = read(f, "mean_squared_error")
    # error = read(f, "rmse")
    error = read(f, "r2")
    mean_rmse = mean(error, dims = 4) # average over stripes
    std_rmse = std(error, dims = 4)
    # unpack
    μ = @view mean_rmse[:, 1, :] # mean rmse
    σ = @view std_rmse[:, 1, :]
    dμ = @view mean_rmse[:, 2, :] # mean rmse of derivative
    dσ = @view std_rmse[:, 2, :]

    nsample, npolicies = read(f, "nsample"), read(f, "npolicies")
    μ = reshape(μ, nsample, npolicies)
    dμ = reshape(dμ, nsample, npolicies)
    σ = reshape(σ, nsample, npolicies)
    # for each kernel, use length scale with best performance
    # lind = zeros(Int, npolicies)
    # for i in 1:npolicies
    #     A = @view dμ[end, i, :]
    #     # lind[i] = argmin(A)
    #     lind[i] = argmax(A)
    # end
    A = zeros(nsample, npolicies)
    B = zeros(nsample, npolicies)
    for i in 1:npolicies
        @. A[:, i] = dμ[:, i] #, lind[i]]
        @. B[:, i] = dσ[:, i] #, lind[i]]
    end
    n = size(error, 5)
    confidence = maximum(B) / sqrt(n)
    println("confidence = $confidence")
    names = read(f, "policy names")
    plot(xlabel = "# of points", ylabel = "R2") #, yscale = :log10)
    plot!(A, label = permutedims(names), legend = :bottomleft, linewidth = 2) #, linestyle = :auto)
    gui()
end

path = "SARA/NatCom2020/results/"
# file = "inner_acquisition_benchmark_iSARA_short.h5"
# file = "inner_acquisition_benchmark_EQ_short.h5"
file = "inner_acquisition_benchmark_EQ_long.h5"

f = h5open(path * file)

# plotly()
# plotlyjs()
using Plots
pyplot()
# Plots.scalefontsizes(1.5)
plot_acquisition_benchmark(f)
# savefig("inner_kernel_plot.pdf")
