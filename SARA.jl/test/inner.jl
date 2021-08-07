module TestInner
using LinearAlgebra
using SARA
using Test
using GaussianDistributions: mean, std

@testset "inner loop" begin
    n = 8
    x = randn(n)
    sort!(x)
    σ = .05
    f(x) = cos(π*x)
    y = f.(x)
    @. y += σ*randn()

    domain = (-2, 2)
    next, posteriors, uncertainty = inner(x, [y], domain, σ)

    for i in eachindex(posteriors)
        p = posteriors[i]
        p_mean, p_std = evaluate(p, x)
        @test isapprox(p_mean, y, atol = 3σ)
        @test all(<(3σ), p_std)
    end

    # using Plots
    # xs = range(-3, 3, length = 128)
    # plot(xs, f.(xs))
    # scatter!(x, y)
    # for i in eachindex(posteriors)
    #     p = posteriors[i]
    #     μ_xs, σ_xs = evaluate(p, xs, normalization[i])
    #     plot!(xs, μ_xs, ribbon = σ_xs)
    # end
    # pnext = evaluate(posteriors[1], next, normalization[1])
    # scatter!(next, pnext) # label = "next point")
    # gui()
end

end
