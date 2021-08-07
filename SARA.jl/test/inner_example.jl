using LinearAlgebra
using GaussianDistributions
using CovarianceFunctions
const Kernel = CovarianceFunctions
using Kernel
using Kernel: gramian
using GaussianDistributions: ConditionalMean, ConditionalKernel
using WoodburyIdentity
using SARA: random_sampling, uncertainty_sampling, inner_sampling, integrated_uncertainty_sampling, ucb, integrated_uncertainty
using ForwardDiff

# C = C | (xi, yi, σ^2) # TODO: make efficient by getting rid of hierarchical structure
function integrated_deviation(f, g, x::AbstractVector, w::AbstractVector = fill(1/length(x), length(x)))
    h(x) = (f(x) - g(x))^2
    return sqrt(integral(h, x, w))
end
function integral(f, x::AbstractVector, w::AbstractVector = fill(1/length(x), length(x)))
    dot(w, f.(x))
end

function sampling_path(policy, G::Gaussian, z::AbstractVector, σ::Real, nsample::Int, ground_truth)
    xs = collect(z) # potential points and integration nodes (z)
    C = G
    x, y = zeros(nsample), zeros(nsample)
    d, dd, u, du = zeros(nsample), zeros(nsample), zeros(nsample), zeros(nsample)
    for i in 1:nsample # random sampling
        xi = policy(C, xs)
        println(xi)
        x[i] = xi
        y[i] = g(xi) # noisy observation
        xs = deleteat!(xs, findfirst(==(xi), xs))
        C = G | @views (x[1:i], y[1:i], σ^2)

        # d[i] = mean((mean(C, z) .- ground_truth).^2)
        u[i] = integrated_uncertainty(C, z)
        D = GaussianDistributions.gradient(C) # multi-ouput process
        d[i] = mean((mean(D[2], z) .- ground_truth).^2)
        du[i] = integrated_uncertainty(D[2], z)
    end
    return C, d, u, du
end

# synthetic example for inner loop acquisition
l = 1/4
k = Kernel.Lengthscale(Kernel.EQ(), l)
G = Gaussian(k)

f(x::Real) = sin(2π*x)
σ = 1e-1
g(x::Real) = f(x) + σ*randn() # noisy observations

# grid of potential points
n = 512
z = range(-1, 1, length = n)
# ground_truth = f.(z)
ground_truth = (x->ForwardDiff.derivative(f, x)).(z)

nsample = 32
sample_helper(policy) = sampling_path(policy, G, z, σ, nsample, ground_truth)
println("random")
Cr, dr, ur, dur = sample_helper(random_sampling)

println("uncertainty")
Cu, du, uu, duu = sample_helper(uncertainty_sampling)

println("integrated uncertainty")
policy = (G, x) -> integrated_uncertainty_sampling(G, x, σ)
Ciu, diu, uiu, duiu = sample_helper(policy)

println("gradient uncertainty")
policy = (G, x) -> inner_sampling(G, x, σ)
Ci, dgu, ugu, dugu = sample_helper(policy)

deviations = (ur, du, diu)
uncertainties = (ur, uu, uiu)

using Plots
plotly()
plot(scale = :log10, legend = :outerbottomleft)
plot!(dr, label = "random")
plot!(du, label = "uncertainty")
plot!(diu, label = "integrated uncertainty")
plot!(dgu, label = "gradient uncertainty")
gui()

# plot(yscale = :log10)
# plot!(ur, label = "random")
# plot!(uu, label = "uncertainty")
# plot!(uiu, label = "integrated uncertainty")
# plot!(ui, label = "gradient uncertainty")
# gui()
#
plot(yscale = :log10)
plot!(dur, label = "random")
plot!(duu, label = "uncertainty")
plot!(duiu, label = "integrated uncertainty")
plot!(dugu, label = "gradient uncertainty")
gui()
