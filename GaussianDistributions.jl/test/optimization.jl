module TestKernelOptimization
using Test
using LinearAlgebra
using GaussianDistributions
using GaussianDistributions: nlml, optimize, optimize!

@testset "kernel optimization" begin
    function k(θ)
        a, l = exp.(θ)
        (x, y) -> a*exp(-(x-y)^2/(2l^2))
    end
    f(x) = sin(x)
    n = 8
    x = randn(n)
    y = f.(x)
    θ = fill(-2., 2) # WARNING: without noise covariance, this has to lead to full rank system
    k0 = k(θ)
    optimize!(k, θ, x, y)
    @test nlml(k(θ), x, y) < nlml(k0, x, y)

    # using Plots
    # G = Gaussian(k(θ))
    # C = G | (x, y)
    # plot(C)
    # scatter!(x, y)
    # gui()

    # with noisy observations
    σ = 5e-2
    @. y += σ * randn()

    θ = fill(-2., 2) # WARNING: without noise covariance, this has to lead to full rank system
    k0 = k(θ)
    θ = optimize(k, θ, x, y, σ^2)
    @test nlml(k(θ), x, y, σ^2) < nlml(k0, x, y, σ^2)

    # G = Gaussian(k(θ))
    # C = G | (x, y, σ^2)
    # plot(C)
    # scatter!(x, y)
    # gui()
end

@testset "noise variance optimization" begin
    l = .5
    k(x, y) = exp(-(x-y)^2/(2l^2))
    f(x) = sin(2π*x)
    n = 128
    x = randn(n)
    y = f.(x)
    σ_truth = 5e-2     # with noisy observations
    @. y += σ_truth * randn()

    σ²_new = .1
    σ²_new = optimize(σ²_new, k, x, y) # optimizing noise variance
    @test nlml(k, x, y, σ²_new) < nlml(k, x, y, σ_truth^2)
    @test isapprox(σ²_new, σ_truth^2, rtol = .5)

    # using Plots
    # G = Gaussian(k)
    # C = G | (x, y, σ²_new)
    # plot(C)
    # scatter!(x, y)
    # gui()
end

end # TestKernelOptimization
