using LinearAlgebra
using CovarianceFunctions
using CovarianceFunctions: AbstractKernel, gramian

# first, let's define the Umlaut Unau CovarianceFunctions
struct UmlautUnau{T, K} <: AbstractKernel{T}
    k::K # Umlaut Unaus can be found wrapping arbitrary kernels k
    sleepiness::Int
end
const UU = UmlautUnau # defining alias
const 🦥 = UU # I you know you love it
function UmlautUnau(k::AbstractKernel = CovarianceFunctions.EQ(), sleepiness::Int = 2) # constructor
    println("🦥: I äm an Umlaut Unau wrapping a bränch of type $(typeof(k)).")
    UU{eltype(k), typeof(k)}(k, sleepiness)
end

# Unfortunately, Unaus are very slow at evaluating things,
# redefining the term "lazy evaluation"
function (k::UU)(x, y)
    if !iszero(k.sleepiness)
        println("🦥: I äm going to take my swëet time evaluating thïs")
        sleep(k.sleepiness)
        println("🦥: Ok, here you gö!")
    end
    k.k(x, y) # makes UU a functor: k(x, y), passes to wrapped CovarianceFunctions
end

# extending factorization
function LinearAlgebra.factorize(G::CovarianceFunctions.Gramian{<:Real, <:🦥}; tol = 1e-6)
    println("🦥: I need to näp beföre fäctörizing!")
    unau = G.k
    sleep(unau.sleepiness)
    println("🦥: Älright, let's dö this!")
    k = unau.k # base CovarianceFunctions the Unau is wrapping to not make it evaluate things
    G = gramian(k, G.x)
    cholesky(G, Val(true), tol = tol) # the Val(true) calles the pivoted algorithm
end

# extend in-place multiplication (5-argument mul! is recommended)
# computes  α * Ax + β y and stores the result in y
function LinearAlgebra.mul!(y::AbstractVector, C::CholeskyPivoted, x::AbstractVector, α::Real = 1, β::Real = 0)
    z = zero(y)
    ip = invperm(C.p)
    U = @view C.U[:, ip] # creates a view into the array, with reordered indices (no allocations)
    mul!(z, U, x) # z = (L' * x)
    mul!(y, U', z, α, β) # y = α * (L * z) + β * y
    return y
end

# example with tests
using Test
k = CovarianceFunctions.MaternP(2) # twice-differentiable Matern CovarianceFunctions
sleepiness = 0
k = UmlautUnau(k, sleepiness) # being wrapped by the Unau

x, y = randn(2)
k(x, y) # evaluation at random pair of points

n = 16
x = randn(n)
G = CovarianceFunctions.gramian(k, x); # this would be very, very slow if it wasn't lazy

F = factorize(G)
@test F isa CholeskyPivoted

α, β = randn(2)
y = randn(size(x))
z = copy(y)
mul!(y, F, x, α, β)
mul!(z, Matrix(F), x, α, β)
@test y ≈ z # testing against canonical implementation

# let's process some Gaussians
using GaussianDistributions
# 1d
G = Gaussian(k) # creates zero mean GP with covariance CovarianceFunctions k

f(x) = sin(x) + cos(x) # some function
x = randn(3) # data points
y = f.(x) # noiseless observations

C = G | (x, y) # conditional Gaussian process

# plotting is the only thing that doesn't yet bypass the Unaus laziness
using Plots
if k.sleepiness == 0
    plot(C, label = "GP")
    scatter!(x, y, label = "data")
    gui()
end

# 3d
d = 3 # dimensions
G = Gaussian(k) # creates zero mean GP with covariance CovarianceFunctions k
α = randn(d)
f(x) = sin(dot(α, x)) # some function
n = 16 # number of data points
x = [randn(d) for _ in 1:n] # random data points

y = f.(x) # noiseless observations
C = G | (x, y) # conditional Gaussian process

if k.sleepiness == 0
    β = randn(d) # random one-d slice to plot on (not very instructive)
    slice(x::Real) = β * x
    projection(x::AbstractVector) = dot(β, x)
    Cp = C ∘ slice
    xp = projection.(x)
    plot_points = range(extrema(xp)..., length = 128)
    plot(Cp, plot_points, label = "GP slice")
    scatter!(xp, y, label = "data")
    gui()
end
