module TestDensity
using Test
using LinearAlgebra
using GaussianDistributions
using GaussianDistributions: pushforward, nlml
using ForwardDiff
using CovarianceFunctions

@testset "density" begin
    @testset "univariate" begin
        μ, σ = randn(), exp(randn())
        G = Gaussian(μ, σ^2)
        @test nld(G, μ) ≈ log(sqrt(2π)*σ)
        @test exp(-nld(G, μ)) ≈ pdf(G, μ)
        x = randn()
        @test exp(-nld(G, x)) ≈ pdf(G, x)
        @test pdf(G)(x) ≈ pdf(G, x)
        @test nld(G)(x) ≈ nld(G, x)
        @test cdf(G)(μ) ≈ 1/2
        @test cdf(G)(Inf) == 1
        @test cdf(G)(-Inf) == 0
    end

    @testset "multivariate" begin
        d = 3
        μ = randn(d)
        Σ = randn(d, d)
        Σ = Σ'Σ
        G = Gaussian(μ, Σ)
        @test nld(G, μ) ≈ logdet(2π*Σ) / 2
        @test exp(-nld(G, μ)) ≈ pdf(G, μ)
        x = randn(d)
        @test exp(-nld(G, x)) ≈ pdf(G, x)
    end

    @testset "process" begin
        μ(x) = zero(x)
        Σ(θ) = (x, y) -> exp(-(x-y)^2/2exp(2θ))
        G = Gaussian(μ, Σ(0.))
        σ = .1
        n = 3
        x = randn(n)
        y = randn(n)
        @test nlml(G, x, y, σ^2) isa Real
        K = cov(marginal(G, x)) + σ^2*I
        @test nlml(G, x, zero(y), σ^2) ≈ (logdet(K) + length(x) * log(2π)) / 2

        # test gradient
        dΣ(θ) = (x, y) -> ForwardDiff.derivative(z->Σ(z)(x, y), θ)
        θ = 1.
        dk = dΣ(θ)
        dK = CovarianceFunctions.gramian(dk, x)
        val, push = pushforward(nlml, K, y)
        @test val ≈ nlml(K, y)
        @test push(dK) isa Real
    end
end

end
