module TestConditional
using Test
using LinearAlgebra
using GaussianDistributions
using GaussianDistributions: ConditionalMean
using ForwardDiff: derivative

const nsigma = 4

@testset "conditional" begin
    @testset "multivariate" begin
        d = 3
        μ = randn(d)
        Σ = randn(d, d)
        Σ = Σ'Σ
        G = Gaussian(μ, Σ)

        x = randn()
        tol = 1e-13

        C = G | (1, x)
        @test C == conditional(G, 1, x)
        @test mean(C[1]) ≈ x
        @test cov(C[1]) < tol # collapse of uncertainty
        @test all(vc < vg for (vc, vg) in zip(var(C), var(G))) # uniform reduction in uncertainty

        x = randn(2)
        C = G | (1:2, x)
        @test mean(C[1:2]) ≈ x
        @test sum(std(C[1:2])) < tol # collapse of uncertainty

        # play around with posterior
        x = randn()
        σ = .1
        C = G | (1, x, σ^2)
        @test abs(mean(C[1])-x) < abs(mean(G[1])-x) # posterior mean is closer to data than prior mean
        @test cov(C[1]) < cov(G[1])  # posterior uncertainty is smaller than prior
        @test sum(var(C)) < sum(var(G))

        # test conversion of vector of noise variances to diagonal covariance
        n = 2
        x = randn(n)
        σ² = σ^2 * exp.(randn(n))
        C = G | (1:n, x, σ²)
        E = G | (1:n, x, Diagonal(σ²))
        @test C isa Gaussian
        @test mean(C) ≈ mean(E)
        @test cov(C) ≈ cov(E)
    end

    @testset "bayesian linear regression" begin
        # i.e. conditioning on linear operator data
        n, m = 4, 8
        A = randn(n, m)
        σx = 2
        x = σx * randn(m)
        b = A*x

        x = Gaussian(σx^2*I(m))

        # noiseless conditioning
        P = x | (A, b)
        PC = Matrix(cov(P))
        λ = sort!(eigvals(PC))
        @test all(<(1e-12), λ[1:n])
        @test all(>(1e-12), λ[n+1:end])
        @test A*mean(P) ≈ b

        # test iterative conditioning
        Q = x | (A[1:n-1, :], b[1:n-1])
        Q = Q | (A[n, :], b[n])
        @test mean(Q) ≈ mean(P)
        @test cov(Q) ≈ cov(P)

        # with noise
        σ = 1e-6
        y = Gaussian(b, σ^2*I(n))
        P = x | (A, y)

        # n observations only uniquely determine n components
        PC = Hermitian(Matrix(cov(P)))
        λ = sort!(eigvals(PC))
        @test all(<(σ), λ[1:n])
        @test all(>(σ), λ[n+1:end])

        # if we observe a complete set, we can reconstruct the solution
        @test norm(A*mean(P)-mean(y)) < 3σ

        # test iterative conditioning
        Q = x | (A[1:n-1, :], y[1:n-1])
        Q = Q | (A[n, :], y[n])
        @test mean(Q) ≈ mean(P)
        @test cov(Q) ≈ cov(P)
    end

    @testset "process mean" begin
        # create dataset
        G = 4
        x = sort(randn(G))
        f(x) = sin(2π*x)

        l = .1
        k(x, y) = exp(-(x-y)^2/2l^2)
        G = Gaussian(k)
        K = k.(x, x')
        cm = ConditionalMean(G, x, f.(x), K)

        @test marginal(cm, x) ≈ f.(x)

        # test differentibility
        using ForwardDiff: derivative
        g(x) = derivative(cm, x)
    end

    @testset "process" begin
        # create dataset
        n = 16
        x = sort(randn(n))
        f(x) = sin(2π*x)
        y = f.(x)

        # setup Gaussian process
        l = .1
        k(x, y) = exp(-sum(abs2, x-y)/2l^2)
        G = Gaussian(k)
        K = k.(x, x')
        C = G | (x, y) # noiseless conditioning
        @test mean(C, x) ≈ f.(x)
        tol = 10eps(Float64)

        v = var(C, x)
        @test all(v .< tol) # collapse of uncertainty

        # noisy conditioning
        n = 128
        x = sort(randn(n))
        y = f.(x)
        σ = .01
        @. y += σ*randn() # adding noise
        C = G | (x, y, σ^2)
        d = mean(C, x)-f.(x)
        @test all(abs.(d) .< nsigma*σ)
        v = var(C, x)
        @test all(v .< σ^2) # collapse of uncertainty

        # test differentibility
        g(x) = derivative(mean(C), x)
        @test isapprox(g(0), 2π, rtol = .1)

        # TODO: iterative conditioning
        # xs = randn()
        # ys = f(xs)
        # C = C | (xs, ys, σ^2)

        # d = mean(C)(xs) - f(xs)
        # @test abs(d) < nsigma*σ
        # tol = σ
        # v = std(C)(xs)
        # @test v < tol

        @testset "single datapoint conditioning" begin
            x = randn()
            y = f(x)
            C = G | (x, y)
            @test mean(C)(x) ≈ y
            @test std(C)(x) < 1e-12

            # with noise
            y += σ*randn()
            C = G | (x, y, σ^2)
            @test isapprox(mean(C)(x), y, atol = nsigma*σ)
            @test std(C)(x) < nsigma*σ
            @test std(C)(x) > σ/2 # uncertainty does not collapse completely

            # single point conditioning with 2d input
            x = randn(2)
            y = randn()
            C = G | (x, y)
            @test mean(C)(x) ≈ y
            @test std(C)(x) < 1e-12

            # with noise
            y += σ*randn()
            C = G | (x, y, σ^2)
            @test isapprox(mean(C)(x), y, atol = nsigma*σ)
            @test std(C)(x) < nsigma*σ
        end
    end
end

end
