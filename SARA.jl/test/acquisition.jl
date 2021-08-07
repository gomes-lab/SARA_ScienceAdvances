module TestAcquisition
using Test
using LinearAlgebra
using GaussianDistributions
using CovarianceFunctions
const Kernel = CovarianceFunctions
using SARA: ucb, inner_sampling, random_sampling, uncertainty_sampling,
            integrated_uncertainty_sampling


@testset "acquisition" begin
    l = 1/2
    k = Kernel.Lengthscale(Kernel.EQ(), l)
    G = Gaussian(sin, k)
    n = 256

    @testset "random sampling" begin
        x = randn(n)
        @test random_sampling(G, x) isa Real
        @test random_sampling(G, x) in x
    end

    @testset "upper confidence bound" begin
        # test on unconditioned GP
        x = range(0, π, length = n)
        α = 3.
        xi = ucb(G, x, α)
        @test isapprox(xi, π/2, atol = 5e-2)

        # create 2d synthetic data and condition
        l = 1/2
        k = Kernel.Lengthscale(Kernel.EQ(), l)
        G = Gaussian(k)
        f(x) = sum(sin, x)
        x = [randn(2) for _ in 1:n]
        σ = .01
        y = @. f(x) + σ*randn()
        C = G | (x, y, σ^2)

        # test ucb sampling on conditioned process
        ns = 1024
        xs = [randn(2) for _ in 1:ns]

        x0 = ucb(C, xs, 0.) # without uncertainty term, chooses point close to optimum
        @test isapprox(f(x0), 2, atol = 3e-1)

        xi = ucb(C, xs, α) # with uncertainty term,
        @test f(x0) > f(xi) # trades off value with uncertainty
        @test f(xi) > 1 # still chooses xi with moderately large value
    end

    @testset "integrated uncertainty sampling" begin
        l = .1
        k = Kernel.Lengthscale(Kernel.EQ(), l)
        G = Gaussian(k)

        # test on unconditioned GP
        n = 32
        x = range(0, π, length = n)
        σ = 1e-2
        y = @. sin(x) + σ * randn()
        C = G | (x, y, σ^2)

        m = 256
        xs = range(0, π, length = m)

        xi = integrated_uncertainty_sampling(G, xs, σ)
        xk = uncertainty_sampling(G, xs)

        xi = integrated_uncertainty_sampling(C, xs, σ)
        xk = uncertainty_sampling(C, xs)
    end

    @testset "SARA inner loop sampling" begin
        l = .1
        k = Kernel.Lengthscale(Kernel.EQ(), l)
        G = Gaussian(sin, k)

        # test on unconditioned GP
        n = 32
        x = range(0, π, length = n)
        σ = 1e-2
        y = @. sin(x) + σ * randn()
        C = G | (x, y, σ^2)
        m = 256
        xs = range(0, π, length = m)

        xi = inner_sampling(G, xs, σ)
        xi = inner_sampling(C, xs, σ)
        # @test isapprox(xi, π/2, atol = 5e-2)
    end
end

end
