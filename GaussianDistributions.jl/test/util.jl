module TestUtil
using LinearAlgebra
using GaussianDistributions: value_gradient_hessian
using Test

@testset "util" begin
    d = 3
    A = randn(d, d)
    A = A'A
    f(x) = dot(x, A, x) / 2
    x = randn(d)
    fx = value_gradient_hessian(f, x)

    @test length(fx) == d^2 + d + 1
    @test fx[1] ≈ f(x)
    @test fx[2:d+1] ≈ A*x
    @test fx[d+2:end] ≈ vec(A)
end
end
