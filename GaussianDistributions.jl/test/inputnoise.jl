module TestInputNoise
using Test
using LinearAlgebra
using GaussianDistributions
using CovarianceFunctions

@testset "input noise" begin
    # set up GP
    l = 1/4
    k = CovarianceFunctions.Lengthscale(CovarianceFunctions.EQ(), l)
    G = Gaussian(k)

    # set up data
    n = 32 # number of observations
    x = randn(n) # uncorrupted input

    σx = 5e-2 # standard deviation of input noise
    σy = 1e-2 # standard deviation of ouput noise
    εx = σx .* randn(n) # input noise
    εy = σy .* randn(n) # output noise

    f(x) = sin(2π*x)
    y = @. f(x + εx) + εy # output

    C = G | (x, y, σx^2) # canonical conditional distribution
    Ci = G | (x, σx^2, y, σy^2) # NIGP conditional distribution

    xs = 2maximum(abs, x) # outside of data range, where regular posterior mean is constant,
    @test var(Ci)(xs) ≈ var(C)(xs) # NIGP variance is unchanged to regular variance
    xs = 2rand(n) .- 1
    nincrease = sum(var(C, xs) .< var(Ci, xs))
    @test nincrease > n * 3/4 # NIGP tends to increase uncertainty in data-rich regions

    # using Plots
    # plot(Ci, label = "NIGP")
    # plot!(C, label = "GP")
    # scatter!(x + εx, y + εy, label = "data")
    # xs = range(minimum(x) - std(x), maximum(x) + std(x), length = 128)
    # plot!(xs, f, label = "f(x)")
    # gui()
end

end

# plotting uncertainty
# xs = range(-3, 3, length = 256)
# plot(xs, var(Ci), label = "NIGP")
# plot!(xs, var(C), label = "GP")
# scatter!(x + εx, zero(x), label = "data")

# d = var(Ci, xs) .- var(C, xs)
# println([t for t in zip(x, d)])
# i = findfirst(<(0), d)
# println(xs[i])
# println(var(Ci)(xs[i]))
# println(var(C)(xs[i]))
# scatter!([xs[i]], [var(Ci)(xs[i])])
# gui()
