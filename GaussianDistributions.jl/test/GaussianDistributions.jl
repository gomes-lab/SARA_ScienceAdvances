module TestGaussianDistributions
using Test
using LinearAlgebra
using GaussianDistributions

@testset "basic properties" begin
    @testset "univariate" begin
        μ, σ = randn(), exp(randn())
        G = Gaussian(μ, σ^2)
        @test G isa Univariate
        @test mean(G) == μ
        @test cov(G) == σ^2
        @test var(G) == σ^2
        @test std(G) == σ
        # indexing
        @test length(G) == 1
        @test G[1] == G # equality
        @test G[1] ≈ G
    end

    @testset "multivariate" begin
        d = 3
        μ = randn(d)
        Σ = randn(d, d)
        Σ = Σ'Σ
        G = Gaussian(μ, Σ)

        @test G isa Multivariate
        @test mean(G) == μ
        @test cov(G) == Σ
        @test var(G) == diag(Σ)
        @test std(G) == sqrt.(diag(Σ))

        # indexing
        @test length(G) == d
        for i in 1:d
            @test G[i] isa Univariate
            @test G[i] == Gaussian(μ[i], Σ[i, i])
        end
        @test G[1:2] isa Multivariate
        @test G[1:2] == Gaussian(μ[1:2], Σ[1:2, 1:2])

        # convenience constructor for diagonal covariance
        σ² = exp(randn())
        G = Gaussian(μ, σ²)
        @test G isa Multivariate
        @test mean(G) ≈ μ
        @test cov(G) ≈ σ² * I(d)
    end

    @testset "process" begin
        n = 3
        μ(x) = zero(x)
        Σ(x, y) = exp(-(x-y)^2/2)
        G = Gaussian(μ, Σ)
        @test !(G isa Univariate)
        @test !(G isa Multivariate)
        @test mean(G) == μ
        @test cov(G) == Σ

        x = randn()
        @test var(G)(x) == Σ(x, x)
        @test std(G)(x) == sqrt(Σ(x, x))

        Gx = G(x)
        @test Gx isa Univariate
        @test mean(Gx) == μ(x)
        @test var(Gx) == Σ(x, x)

        x = randn(3)
        Gx = marginal(G, x)
        @test Gx isa Multivariate
        @test mean(Gx) == μ.(x)
        @test cov(Gx) == Σ.(x, x')
        @test std(Gx) == sqrt.(Σ.(x, x))
    end
end

end
