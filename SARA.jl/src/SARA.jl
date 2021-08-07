module SARA
using JSON
using Base.Threads
using LinearAlgebra
using Interpolations
using ForwardDiff

using CovarianceFunctions
const Kernel = CovarianceFunctions
using GaussianDistributions
using GaussianDistributions: input_transformation

export innerloop, inner, uncertainty_sampling, evaluate, TemperatureProfile,
        next_stripe, get_relevant_T, iSARA, oSARA, xrd_to_global, stripe_to_global,
        stripe_uncertainty_sampling

include("util.jl")
include("estimate_noise.jl")
include("temperature_profile.jl")
include("acquisition.jl") # acquisition functions
include("inner.jl") # inner and outer loops for optical spectroscopy
include("stripe_to_global.jl")
include("outer.jl")

end # SARA
