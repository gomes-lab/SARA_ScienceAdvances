module TestSample
using Test
using LinearAlgebra
using LinearAlgebraExtensions: LowRank
using WoodburyIdentity
using GaussianDistributions
# using GaussianDistributions: sample, Gaussian
# using Statistics: mean, cov, std, var

@testset "sample" begin
    n, k = 3, 1
    σ = .1
    U = σ*randn(n, k)
    L = LowRank(U)
    A = Matrix(L)
    NL = Gaussian(L)
    NA = Gaussian(A)
    # after 128 samples, this test expects the empirical estimates to be
    # within 5σ of the truth, not fail-proof but highly probable
    function sample_test(N::Gaussian, n = 128; atol = 5σ*sqrt(length(N))/sqrt(n))
        S = sample(N, n, tol = 1e-12)
        @test isapprox(mean(S), mean(N), atol = atol)
        @test isapprox(cov(S), Matrix(cov(N)), atol = atol)
    end
    m = 4 # number of samples
    s = sample(NL, m)
    @test s isa Vector{<:AbstractVector}
    @test length(s) == m
    @test length(s[1]) == n
    sample_test(NL)
    sample_test(NA)

    s = sample(NL, tol = 1e-8)
    @test length(s) == n
    @test s[1] isa Real
    s = sample(NA, tol = 1e-8)
    @test length(s) == n
    @test s[1] isa Real

    # sampling from Woodbury covariance
    W = Woodbury(1e-3*I(n), L)
    μ = randn(n)
    NW = Gaussian(μ, W)
    s = sample(NW, tol = 1e-8)
    @test length(s) == n
    sample_test(NW)
end

end
