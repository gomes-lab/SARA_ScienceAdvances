################### acquisition functions for active learning ##################
random_sampling(G::Gaussian, x::AbstractVector) = rand(x)
random_sampling(G::Gaussian, x::AbstractMatrix) = rand(eachcol(x))

# choose the x with the highest uncertainty, according to G
function uncertainty_sampling(G::Gaussian, x::AbstractVector)
    σ = zeros(length(x))
    @threads for i in eachindex(x)
        σ[i] = std(G)(x[i])
    end
    i = argmax(σ)
    xs = x[i]
    return xs
end

# returns integrated uncertainty at integration nodes x with weights w
function integrated_uncertainty(G::Gaussian, x::AbstractVector, w::AbstractVector = fill(1/length(x), length(x)))
    dot(w, var(G, x))
end

function integrated_uncertainty_sampling(σ::Real)
    (G, x) -> integrated_uncertainty_sampling(G, x, σ)
end

# could introduce weights
function integrated_uncertainty_sampling(G::Gaussian, x::AbstractVector, σ::Real, w = ones(size(x)))
    n = length(x)
    K = Kernel.gramian(cov(G), x)
    K = Matrix(K) # this is the bottleneck
    u = zeros(n) # posterior uncertainty
    for (i, k) in enumerate(eachcol(K))
        u[i] = -sum(abs2, k) / (K[i, i] + σ^2) # technically have to add tr(K), but it doesn't change the minimum
    end
    return x[argmin(u)]
end

# ternary mul
using WoodburyIdentity
function Base.:*(x::AbstractMatrix{<:Real}, W::Woodbury, y::AbstractVecOrMat)
    x*(W*y)
    # xU = x*W.U
    # Vy = (x ≡ y' && W.U ≡ W.V') ? xU' : W.V*y
    # *(x, W.A, y) + W.α * *(xU, W.C, Vy) # can avoid two temporaries
end

############### acquisition function for SARA's inner loop #####################
# picks one-step Bayes optimal point to minimize derivative uncertainty
function inner_sampling(G::Gaussian, x::AbstractVector, σ::Real)
    n = length(x)
    D = GaussianDistributions.value_derivative(G) # multi-ouput process
    K = Kernel.gramian(cov(D), x)
    K11, K22, K21 = zeros(n, n), zeros(n, n), zeros(n, n)
    @threads for i in 1:n # derivative w.r.t. first component
        for j in 1:n
            kij = K.A[i, j] # ijth block
            K11[i, j] = kij[1, 1]
            K21[i, j] = kij[2, 1]
            K22[i, j] = kij[2, 2]
        end
    end
    u = zeros(size(x)) # uncertainty in derivative
    for i in 1:n
        u[i] = - sum(abs2, @view K21[:, i]) / (K11[i, i] + σ^2) # + K22
    end
    return x[argmin(u)]
end
inner_sampling(σ::Real) = (G, x) -> inner_sampling(G, x, σ)

################ upper-confidence-bound sampling ###############################
ucb_acquisition(G::Gaussian, α::Real) = x -> mean(G)(x) + α * std(G)(x)
ucb_acquisition(α::Real) = (G::Gaussian, x) -> ucb_acquisition(G, α)(x)

# choose a point among x with highest confidence bound score
# if α > 0 we are sample at points with high uncertainty
# if α = 0 it reduces to optimization of the predictive mean
# the larger α > 0, the more we exploit known regions with high function value
function ucb(G::Gaussian, x::AbstractVector, α::Real)
    a = ucb_acquisition(G, α)
    score = zeros(length(x))
    @threads for i in eachindex(x)
        score[i] = a(x[i])
    end
    i = argmax(score)
    return x[i]
end
ucb_sampling(α) = (G, x) -> ucb(G, x, α)

# optimizes x
# function ucb(G::Gaussian, x, α::Real, optimize::Val{true})
#     score(x) = mean(M)(x) + α * std(M)(x)
#     # IDEA: maximize score with respect to x
#     return -1
# end

####################### outer loop sampling strategies #########################
# modified sampling policy, taking into account that SARA
# is measuring entire lines in T (stripes) through lateral gradient LSA.
# G is result of get_gradient_map calculated from inner loop data,
# x is a vector of potential (T_max, log10_τ)-conditions
# returns most promising condition
function stripe_sampling(G::Gaussian, x::AbstractVector{<:Tuple}, relevant_T, acquisition)
    u = zeros(length(x)) # acquisition value
    @threads for i in eachindex(x)
        T_max, log10_τ = x[i]
        x_other = x[i][3:end] # other dimensions (e.g. composition)
        T = relevant_T(T_max, log10_τ)
        ci = tuple.(T, log10_τ, x_other...)
        u[i] = sum(x->acquisition(G, x), ci) # IDEA: could further parallize this sum
    end
    return x[argmax(u)]
end

# if X is a matrix, interpret columns as conditions to be measured NOT T peak conditions
function stripe_sampling(G::Gaussian, X::AbstractMatrix{<:Tuple}, acquisition)
    n = size(X, 2)
    u = zeros(n) # acquisition value
    @threads for i in 1:n
        xi = @view X[:, i]
        u[i] = sum(x->acquisition(G, x), xi) # IDEA: could further parallize this sum
    end
    return x[argmax(u)]
end

# returns outer_sampling acquisition function
# relevant_T calculates the relevant temperatures for each stripe condition
function stripe_sampling(relevant_T, acquisition)
    function (G::Gaussian, conditions::AbstractVector{<:Tuple})
        stripe_sampling(G, conditions, relevant_T, acquisition)
    end
end
# uncertainty and upper confidence bound sampling generalized to stripes
stripe_uncertainty_sampling(relevant_T) = stripe_sampling(relevant_T, (G, x)->var(G)(x))
function stripe_ucb_sampling(relevant_T, α::Real)
    stripe_sampling(relevant_T, ucb_acquisition(α))
end

################################################################################
# relevant temperatures with constant offset from T_max
function get_relevant_T(T_max::Real, log10_τ::Real, constant::Val{true},
                        c_low::Real, c_high::Real, n::Int,)
    collect(range(T_max - c_low, T_max - c_high, length = n))
end

# relevant temperatures with proportional offset from T_max
function get_relevant_T(T_max::Real, log10_τ::Real, constant::Val{false},
                        p_min::Real, p_max::Real, n::Int)
    collect(T_max * range(p_min, p_max, length = n))
end

# lazifies evaluation on T_max and log10_τ
function get_relevant_T(constant::Union{Val{true}, Val{false}}, c_min::Real, c_max::Real, n::Int)
    get_T(T_max::Real, log10_τ::Real) = get_relevant_T(T_max, log10_τ, constant, c_min, c_max, n)
end

# makes constant offset the default
function get_relevant_T(c_min::Real, c_max::Real, n::Int)
    constant = Val(true)
    get_T(T_max::Real, log10_τ::Real) = get_relevant_T(T_max, log10_τ, constant, c_min, c_max, n)
end
