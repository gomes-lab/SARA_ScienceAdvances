module TestInterface
using Plots
plotly()
using GaussianDistributions
using SARA
############################ Loading Data  #####################################

path = "/Users/sebastianament/Documents/SEA/XRD Analysis/SARA/Bi2O3_19F44_01/"
log10_τ, T, center_real, pos_wafer, coeffs = SARA.load_data(path)
num_coeffs, num_pos, num_stripes = size(coeffs)

###########################
position = pos_wafer[:, 1] .- center_real[1]
optical = [coeffs[c,:,1] for c in 1:size(coeffs, 1)]

# subsample
using Random: randperm
n = 8
rand_ind = randperm(num_pos)[1:n]
sub_pos = position[rand_ind]
sub_opt = [optical[i][rand_ind] for i in 1:num_coeffs]

# pre-process
domain = (minimum(position), maximum(position))
############################## run inner loop ##################################
c = 1/2
μ = 0.
σ = .25
β = 4.
parameters = [(c, μ, σ, β)]
rescale = SARA.get_rescale(parameters)
@time next, posteriors, max_uncertainty =
                        innerloop(sub_pos, sub_opt, domain;
                            grid = position, rescale_parameters = parameters)
# ################################ plotting ######################################
xs = position
for i in 1:1 #eachindex(posteriors)
    mu, std = evaluate(posteriors[i], xs)
    plot(position, optical[i], label = "complete data", legend = :bottomright)
    scatter!(sub_pos, sub_opt[i], label = "measured data")
    plot!(xs, mu, ribbon = 2std, label = "GP")
    scatter!(next, zero(next), label = "next")
    gui()
end

using GaussianDistributions: sample, mean, cov
using LinearAlgebra
# P = posteriors[1]
# M = marginal(P, xs)
# M = Gaussian(mean(M), Symmetric(cov(M)))
# tol = 1e-12
# factorize?
# K = cholesky(Symmetric(cov(M)), Val(true), check = false)
# P = cholesky(P)
# plot!(xs, sample(M, 3))
# gui()

######################## convert to T ###############################
ind = sortperm(T[:, 1])
temperature = T[ind, 1]/maximum(T[:,1])
position = position[ind]
domain = (minimum(temperature), maximum(temperature))
DT, UT = SARA.get_global_data(position, temperature, temperature, posteriors, domain)

########################## gradient map ##############################
n = 128
domain_scale = 1
temp = domain_scale * rand(n)
dwell = domain_scale * rand(n)
# create synthetic gradient map (two stripes)
∇(x) = exp(-((x[1]-.25)/.05)^2/2) + exp(-((x[1]-.75)/.05)^2/2)

# create synthetic gradient map (donut)
# m = 0.5
# s = .1
# ∇(x) = exp(-(norm(x.-fill(.5, length(x)))-m)^2/2s^2)

DT = ∇.(tuple.(temp, dwell))
UT = 1e-2exp.(randn(size(DT))) # non-uniform noise

grid = 0:1/50:1
grid = tuple.(grid, grid')
temperature_grid = [g[1] for g in grid]
dwelltime_grid = [g[2] for g in grid] # TODO: lazy grid?

domain = ((0., domain_scale * 1.), (0., domain_scale * 1.))
gradient_map = SARA.get_gradient_map(temp, dwell, DT, UT, domain)

M, S = SARA.evaluate_gradient_map(gradient_map, temperature_grid,
                                    dwelltime_grid, domain)

########################## outer loop ####################################
next_temp, next_dwell = SARA.outerloop(temp, dwell, gradient_map, temperature_grid[:], dwelltime_grid[:],
                                        domain, num_next = 1)
temp, dwell = 0:1/50:1, 0:1/50:1
next = (next_dwell, next_temp)
heatmap(dwell, temp, M)
scatter!([next])
gui()

end # TestSARA
