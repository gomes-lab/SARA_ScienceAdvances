module TestMulti
using Test
using LinearAlgebra
using GaussianDistributions
using GaussianDistributions: gradient, integral, input_transformation, shift, scale,
            value_derivative, value_gradient, value_gradient_hessian
using ForwardDiff, CovarianceFunctions

@testset "VectorGaussian" begin
    # @testset "derivative process" begin
    #     l = .5
    #     k = (x, y) -> exp(-sum(abs2, x-y)/(2l^2))
    #     G = Gaussian(sin, k)
    #     x, y = randn(2)
    #     # testing gradient process with 1d input
    #     D = value_derivative(G)
    #     x = randn()
    #     gx = D(x)
    #
    #     @test mean(gx) ≈ [sin(x), cos(x)]
    #     @test size(cov(gx)) == (2, 2)
    #     @test cov(gx)[1, 1] ≈ k(x, x)
    #
    #     # testing gradient with 2d input
    #     d = 3
    #     f(x::AbstractVector) = sum(sin, x)
    #     G = Gaussian(f, k)
    #     D = value_gradient(G, d)
    #     x = randn(d)
    #
    #     Dx = D(x)
    #     @test mean(Dx) ≈ [f(x), cos.(x)...]
    #     @test size(cov(Dx)) == (d+1, d+1)
    #
    #     # testing output indexing for multi-output GP
    #     H = D[1] # first dimension of gradient GP is equal to original GP
    #     x, y = randn(d), randn(d)
    #     @test mean(H)(x) ≈ mean(G)(x)
    #     @test cov(H)(x, y) ≈ cov(G)(x, y)
    #
    #     G = Gaussian(f, k)
    #     D = value_gradient_hessian(G, d)
    #     x = randn(d)
    #
    #     Dx = D(x)
    #     dd = d^2 + d + 1
    #     @test size(cov(Dx)) == (dd, dd)
    #     @test mean(D)(x) ≈ value_gradient_hessian(f, x)
    # end

    @testset "vector conditional" begin
        d = 3
        A = randn(d, d)
        A = A'A
        f(x) = dot(x, A, x)/2
        g(x) = ForwardDiff.gradient(f, x)

        x = randn(d)

        k = CovarianceFunctions.EQ()
        G = gradient(Gaussian(k), d)

        y = g(x)
        C = G | ([x], [y]) # noiseless conditioning

        @test mean(C)(x) ≈ y
        @test var(C)(x) ≈ zeros(d)

        z1, z2 = randn(d), randn(d)
        C11 = cov(C)(z1, z1)
        C12 = cov(C)(z1, z2)
        tol = 1e-6
        for i in 1:d, j in 1:d
            @test abs(C11[i, j] - cov(C)[i, j](z1, z1)) < tol # same point
            @test abs(C12[i, j] - cov(C)[i, j](z1, z2)) < tol
        end

        σ = 0.01
        C = G | ([x], [y], σ^2) # noisy conditioning
        @test isapprox(mean(C)(x), y, atol = 3σ)
        @test !all(==(0), var(C)(x))
        @test all(<(σ^2), var(C)(x))
    end
end

end
