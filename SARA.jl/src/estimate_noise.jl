using CovarianceFunctions
const Kernel = CovarianceFunctions
using GaussianDistributions
using Statistics
using GaussianDistributions: optimize

# get estimate of sigma
function estimate_noise(x, y, k = Kernel.Lengthscale(Kernel.MaternP(2), .2))
    σ = 0.05 # initial guess
    σ² = optimize(σ^2, k, x, y, tol = 1e-6) # optimizing marginal likelihood
    sqrt(σ²)
end
