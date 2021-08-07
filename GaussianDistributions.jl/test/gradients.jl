using GaussianDistributions
using CovarianceFunctions: GradientKernel, ValueGradientKernel

k = CovarianceFunctions.EQ()
g = GradientKernel(k)
