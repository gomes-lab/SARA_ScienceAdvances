using HDF5

# first, plot acquisition benchmark with input-noise-corrected data
path = "SARA/NatCom2020/outer/results/"
f = h5open(path * "outer_acquisition_benchmark.h5")
nrandom = read(f, "nrandom")
r2 = read(f, "r2")
close(f)

f = h5open(path * "outer_acquisition_benchmark_input_noise_0.h5")
r2_no_input_noise = read(f, "r2")
close(f)

using Statistics
function random_statistics(x::AbstractMatrix, nrandom::Int)
    random_statistics(@view(x[:, 1:nrandom]))
end
function random_statistics(x::AbstractMatrix)
    m = vec(mean(x, dims = 2))
    s = vec(std(x, dims = 2))
    return m, s
end

function plot_results(r2::AbstractMatrix, nrandom::Int; linestyle = :solid)
    # decide how many iterations to plot
    nsample = 64
    r2 = @view r2[1:nsample, :]
    r2_random, r2_random_std = random_statistics(r2, nrandom)

    r2 = @view r2[:, nrandom+1:end]
    r2_sigma = r2[:, 1]
    r2_stripe_sigma = r2[:, 2]
    nucb = (size(r2, 2)-3) ÷ 2
    r2_ucb = r2[:, 3:2+nucb]
    r2_stripe_ucb = r2[:, 3+nucb:end]

    # choose the ucb parameter that is best at iteration ind
    ind = nsample ÷ 2
    k = argmax(r2_ucb[ind, :])
    r2_ucb = r2_ucb[:, k]
    k = argmax(r2_stripe_ucb[ind, :])
    r2_stripe_ucb = r2_stripe_ucb[:, k]

    p!(y, label, color) = plot!(y, label = label, linestyle = linestyle, linecolor = color)
    data = [r2_random, r2_sigma, r2_stripe_sigma, r2_ucb, r2_stripe_ucb]
    labels = ["random", "sigma", "stripe sigma", "ucb", "stripe ucb"]
    colors = palette(:seaborn_muted)[1:length(data)]
    p!.(data, labels, colors)
end

using Plots
plotly()
plot(legend = :bottomright, linewidth = 1.,
     xlabel = "number of samples", ylabel = "R²ₛ score",
     framealpha = 0, foreground_color_legend = nothing, background_color_legend = nothing,  grid=true, framestyle = :box)
ylims!((0, .8)) # .8 is approximately the converged r2 value
plot_results(r2, nrandom)
plot_results(r2_no_input_noise, nrandom, linestyle = :dash)
gui()
