# returns vector of function value and gradient, using diff results
function value_gradient(f, x::AbstractVector, diffresults::Val{true})#  = Val(true))
    r = DiffResults.GradientResult(x) # this could take pre-allocated temporary storage
    ForwardDiff.gradient!(r, f, x)
    return vcat(r.value, r.derivs[1]) # d+1 # ... to avoid this
end
function value_gradient(f, x::AbstractVector, diffresults::Val{false} = Val(false))
    ∇ = ForwardDiff.gradient(f, x)
    return vcat(f(x), ∇)
end

function value_gradient_hessian(f, x::AbstractVector)
    d = length(x)
    dd = d^2 + d + 1
    fx = similar(x, dd)
    fx[1] = f(x)
    grad = @view fx[2:d+1]
    hess = @view fx[d+2:end]
    results = DiffResults.DiffResult(grad, hess)
    g(x) = ForwardDiff.gradient(f, x)
    ForwardDiff.jacobian!(results, g, x)
    # IDEA: try to minimize evaluations to a single pass
    # hessian = reshape(hessian, d, d) # needs to be d + 1  by d
    # result = DiffResults.DiffResult(valgrad, hessian)
    # println(result)
    # ForwardDiff.jacobian!(result, x->value_gradient(f, x), x)
    return fx

end

function value_derivative(f, x::Real, diffresults::Val{true} = Val(true))
    r = DiffResults.DiffResult(zero(x), zero(x)) # this could take pre-allocated temporary storage
    r = ForwardDiff.derivative!(r, f, x)
    return vcat(r.value, r.derivs[1])
end
function value_derivative(f, x::Real, diffresults::Val{false})
    ∇ = ForwardDiff.derivative(f, x)
    return vcat(f(x), ∇)
end

################################################################################
issquare(A::AbstractMatrix) = size(A, 1) == size(A, 2)
issquare(A::Real) = true
# converts y in various forms to a vector of numbers
_vec(Y::AbstractMatrix) = vec(Y)
_vec(y::AbstractVector{<:Number}) = y
# this needs some thinking if we want to make static arrays efficient
function _vec(y::VecOfVec{<:Number}) # different behavior to vec!
	d = length(y[1])
	u = zeros(length(y)*d)
	for i in eachindex(y)
		u[d*(i-1)+1:d*i] .= y[i]
	end
	return u
end

########################### Laplace Approximation ##############################
# IDEA: expand this into variational approximations (mean field etc.)
# IDEA: connect with GP classification
# ForwardDiff.hessian(f, x::Real) = FD.derivative(y->FD.derivative(f, y), x)
# approximates a pdf f with a normal distribution centered at x using laplace's method
# TODO: height of Gaussian?
function laplace(f, x::AbstractVector)
    H = ForwardDiff.hessian(y->log(f(y)), x)
    Gaussian(x, inverse(H))
end
