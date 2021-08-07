module TestLinear
using Test
using LinearAlgebra
using GaussianDistributions
using GaussianDistributions: input_transformation, shift, scale, integral, gradient, value_gradient
using CovarianceFunctions
using ForwardDiff

function algebra_test(G, a, b)
    μ = mean(G)
    H = G + b
    @test mean(H) ≈ μ + b
    @test cov(H) ≈ cov(G)

    H = b + G
    @test mean(H) ≈ μ + b
    @test cov(H) ≈ cov(G)

    H = G - b
    @test mean(H) ≈ μ - b
    @test cov(H) ≈ cov(G)

    H = b - G
    @test mean(H) ≈ b - μ
    @test cov(H) ≈ cov(G)

    H = a*G
    @test mean(H) ≈ a*μ
    @test cov(H) ≈ a*cov(G)*a

    H = G*a
    @test mean(H) ≈ a*μ
    @test cov(H) ≈ a*cov(G)*a

    H = G / a
    @test mean(H) ≈ μ / a
    @test cov(H) ≈ cov(G)/a^2

    H = a \ G
    @test mean(H) ≈ μ / a
    @test cov(H) ≈ cov(G)/a^2
end

# testing linear functionals of Gaussian distributions
@testset "linear transformations" begin
    @testset "univariate" begin
        μ, σ = randn(2)
        G = Gaussian(μ, σ^2)

        a, b = randn(2)
        algebra_test(G, a, b)
    end

    @testset "multivariate" begin
        n = 3
        μ = randn(n)
        Σ = randn(n, n)
        Σ = Σ'Σ
        G = Gaussian(μ, Σ)

        a = randn()
        b = randn(n)
        algebra_test(G, a, b)

        A = randn(2, n) # linear transformation
        H = A*G
        @test mean(H) ≈ A*μ
        @test cov(H) ≈ A*cov(G)*A'

        w = randn(n)
        @test dot(w, G) isa Univariate
        @test mean(dot(w, G)) ≈ dot(w, mean(G))
        @test cov(dot(w, G)) ≈ *(w', cov(G), w)

        @test sum(G) isa Univariate
        @test diff(G) isa Multivariate
        @test length(diff(G)) == length(G)-1
    end

    @testset "process" begin
        l = .5
        k = (x, y) -> exp(-sum(abs2, x-y)/(2l^2))
        G = Gaussian(sin, k)
        # testing scaling of GP
        a, b = randn(2)
        H = a*G
        @test H isa Gaussian
        x, y = randn(2)
        @test mean(H)(x) ≈ a * sin(x)
        @test cov(H)(x, y) ≈ a^2 * k(x, y)

        # testing division by scalar
        H = G / b
        @test H isa Gaussian
        @test mean(H)(x) ≈ sin(x) / b
        @test cov(H)(x, y) ≈ k(x, y) / b^2

        # testing addition to GP
        H = G + b
        @test H isa Gaussian
        @test mean(H)(x) ≈ sin(x) + b
        @test cov(H)(x, y) ≈ k(x, y)
    end

    @testset "input transformation" begin
        l = .5
        k = (x, y) -> exp(-sum(abs2, x-y)/(2l^2))
        G = Gaussian(sin, k)
        x, y = randn(2)

        # general input transformation
        f(x) = exp(-x^2)
        H = input_transformation(G, f)
        @test H(x) isa Gaussian
        @test mean(H(x)) ≈ mean(G(f(x)))
        @test cov(H)(x, y) ≈ cov(G)(f(x), f(y))

        # composition syntax
        H = G ∘ f
        @test H(x) isa Gaussian
        @test mean(H(x)) ≈ mean(G(f(x)))
        @test cov(H)(x, y) ≈ cov(G)(f(x), f(y))

        # shifting and scaling
        a, b = randn(2)
        H = shift(G, b)
        @test mean(H(x)) ≈ mean(G(x+b))
        @test cov(H)(x, y) ≈ cov(G)(x+b, y+b)

        H = scale(G, a)
        @test mean(H(x)) ≈ mean(G(a*x))
        @test cov(H)(x, y) ≈ cov(G)(a*x, a*y)
    end

    @testset "integral" begin
        l = .5
        k = (x, y) -> exp(-sum(abs2, x-y)/(2l^2))
        # testing 1d integral
        G = Gaussian(k)
        S = integral(G, 0, 2π)
        @test S isa Gaussian
        @test mean(S) == 0
        @test std(S) < 1

        # testing integral on conditional process
        n = 64 # enough points to determine process to high accuracy
        x = 2π * rand(n)
        σ = 1e-2
        y = @. sin(x) + σ*randn()
        C = G | (x, y, σ^2)
        S = integral(C, 0, 2π)
        @test isapprox(mean(S), 0, atol = σ)
        @test std(S) < 2σ
    end

    @testset "differential" begin
        k = CovarianceFunctions.RQ(1.)
        G = Gaussian(k)
        f = x->sum(sin, x)
        d, n = 3, 5
        X = randn(d, n)
        x = [c for c in eachcol(X)]
        y = f.(x)
        dy = (z->ForwardDiff.gradient(f, z)).(x)
        C = G | (x, gradient, dy, d)
        @test C isa Gaussian
        tol = 1e-4
        grad_cov = (x, y)->ForwardDiff.jacobian(z2->ForwardDiff.gradient(z1->C.Σ(z1, z2), x), y)
        for i in eachindex(x)
            @test isapprox(ForwardDiff.gradient(C.μ, x[i]), dy[i], atol = tol)
            @test C.Σ(x[i], x[i]) > tol # covariance in value does not collapse without value observations
            GC = grad_cov(x[i], x[i])
            @test norm(GC) < d^2*tol # gradient uncertainty collapses
        end

        ydy = [vcat(yi, dyi) for (yi, dyi) in zip(y, dy)]
        C = G | (x, value_gradient, ydy, d)
        @test C isa Gaussian
        tol = 1e-4
        grad_cov = (x, y)->ForwardDiff.jacobian(z2->ForwardDiff.gradient(z1->C.Σ(z1, z2), x), y)
        for i in eachindex(x) # these once failed, possibly due to ill conditioning
            @test isapprox(value_gradient(C.μ, x[i]), ydy[i], atol = tol)
            @test C.Σ(x[i], x[i]) < tol # covariance in value does collapse with value observations
            GC = grad_cov(x[i], x[i])
            @test norm(GC) < d^2*tol # gradient uncertainty collapses
        end

    end
end

end
