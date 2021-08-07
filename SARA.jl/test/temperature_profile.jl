module TestTemperatureProfile
using Test
using SARA
using SARA: TemperatureProfile, inverse_profile

@testset "temperature profile" begin
    P = TemperatureProfile()
    # 1. correctness of inverse profile
    T_max = 800
    log10_τ = 3
    f = P(T_max, log10_τ)
    g = inverse_profile(P, T_max, log10_τ)
    n = 16
    for T in T_max*rand(n)
        @test f(g(T)) ≈ T
    end
    for x in 1e-1randn(n)
        @test g(f(x)) ≈ abs(x)
    end

    T_max = 1025.0
    log10_τ = 4.0
    P = TemperatureProfile()
    f = P(T_max, log10_τ) # maps position of measured optical data to temperature
    g = inverse_profile(P, T_max, log10_τ)
    T = T_max / 2
    @test T ≈ f(g(T))
    x = 1e-1randn()
    @test abs(x) ≈ g(f(x))

    # test differentiation of g
    using ForwardDiff: derivative
    d = derivative(g, T)
    @test d isa Real
end

# d = (t->ForwardDiff.derivative(mean(C), t)).(T) # WARNING: investigate numerical issue (NaN)

end

# using JSON
# 2. testing against Bi2O3 data
# include("../NatCom2020/inner_load_data.jl")
#
# path = "/Users/sebastianament/Documents/SEA/XRD Analysis/SARA/Bi2O3_19F44_01/"
# # path = ""
# file = "Bi2O3_19F44_01_inner_loop_data.json"
# f = JSON.parsefile(path * file)
# position, optical, rescaling_parameters = load_data(f)
#
# tol = 1e-6
# P = TemperatureProfile()
# for i in 1:length(f)
#     x = 1000 * (f[i]["pos_wafer"] .- f[i]["center_real"])
#     T = f[i]["temp_profile"]
#     # T = f[i]["temp_profile_si"]
#     T_max, τ = f[i]["meta"]["temp"], f[i]["meta"]["dwell"]
#     log10_τ = log10(τ)
#     if maximum(abs, P(T_max, log10_τ).(x) .- T) > tol
#         println("Hey!")
#     end
# end

#### old plot of fwhm
# mn = [2.39794001e+00, 4.00000000e+02, 2.50000000e+01, 1.77686356e-02, 0.00000000e+00]
# mx = [   4.,         1300.,          350.,            9.92052851,    5.,        ]
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#plt.xlabel('log tau')
#plt.ylabel('temp')
#X,Y = meshgrid(range(mn[0], mx[0], 20), range(mn[1], mx[1], 20))

# X = collect(range(mn[1], stop=mx[1], length=20))
# Y = collect(range(mn[2], stop=mx[2], length=20))

# using Plots
# pyplot()
# Z = ffwhm(Y, X)
# surface(X[:], Y[:], ffwhm.(Y, X'), rstride=1, cstride=1, alpha=0.2)
