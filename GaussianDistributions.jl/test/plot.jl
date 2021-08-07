module TestPlot
using GaussianDistributions
using Plots
using Kernel
using Test
@testset "plot" begin
    l = .3
    k = Kernel.Lengthscale(Kernel.EQ(), l)
    G = Gaussian(k)

    n = 8
    σ = 1e-2
    x = randn(n)
    y = @. sin(2π*x) + σ * randn()

    C = G | (x, y, σ^2)
    fig = plot(C)
    scatter!(x, y)
    gui()
    @test fig isa Plots.Plot
end
end
