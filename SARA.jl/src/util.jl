# data loader
function load_data(path)
    file = "legendre_coefficients.json"
    f = JSON.parsefile(path * file)

    λ = get(f, "wavelengths", -1)
    λ = float.(λ)
    spec = get(f, "spectra", -1)

    num_spec = length(spec)
    coeffs_1 = get(spec[1], "legendre_coeffs", -1)
    num_pos = length(coeffs_1)
    num_coeffs = length(coeffs_1[1])

    log10_τ = zeros(num_spec)
    temp = zeros(num_pos, num_spec)
    pos_wafer = zeros(num_pos, num_spec)
    center_real = zeros(num_spec) # center of symmetry
    # dimensions are: coefficients x positions on stripe x spectrogram
    coeffs = zeros(num_coeffs, num_pos, num_spec)
    for i = 1:num_spec
        # log10 dwell time
        log10_τ[i] = get(spec[i], "logtau", -1)

        # temperature profile
        temp[:,i] = float.(get(spec[i], "temp_profile_si", -1))
        # temp[:,i] = float.(get(spec[i], "temp_profile_linear", -1))

        # positions on the wafer
        pos_wafer[:,i] = float.(get(spec[i], "pos_wafer", -1))

        center_real[i] = float(get(spec[i], "center_real", -1))

        # optical coefficients
        coeffs_i = get(spec[i], "legendre_coeffs", -1)
        for j = 1:num_pos
            coeffs[:, j, i] = float.(coeffs_i[j])
        end
    end
    T = temp
    return log10_τ, T, center_real, pos_wafer, coeffs
end

############################ data processing ###################################
# center and normalize the data
function center(x::AbstractVector)
    m, s = mean(x), std(x)
    y = @. (x - m) / s
    return y
end
# returns a linear map from [a, b] to [0, 1]
unit_transform(a, b) = x->(x-a)/(b-a)
inv_unit_transform(a, b) = x -> (b-a)*x + a # inverse unit transform

function normalize_data(data::AbstractArray{<:Real}, domain::NTuple{2, <:Real})
    unit_transform(domain...).(data)
end

######################## dimensionality reduction ##############################
# 2. calculate xrd coefficients based on PCA
using ApproxFun
# calculates Legendre coefficients
function legendre_coefficients(X::AbstractArray, m::Int)
    n = size(X, 1)
    leastsquares! = legendre_coefficients(n, m)
    coefficients = zeros(m, size(X, 2), size(X, 3))
    @threads for i in 1:size(X, 3)
        println("coefficients for stripe $i")
        Xi = @view X[:, :, i]
        Ci = @view coefficients[:, :, i]
        leastsquares!(Ci, Xi)
    end
    coefficients
end
function legendre_coefficients(n::Int, m::Int)
    x = range(-1, 1, length = n)
    A = zeros(n, m)  # Vandermonde matrix
    for k in 1:m
        A[:, k] = Fun(Legendre(), vcat(zeros(k-1), 1)).(x)
    end
    A = qr(A)
    leastsquares(C, X::AbstractMatrix) = ldiv!(C, A, X)
    leastsquares(X::AbstractMatrix) = A \ X
end

############ based on PCA
# n is number of coefficients to calculate
function svd_coefficients(X::AbstractArray, n::Int)
    C = zeros(n, size(X, 2), size(X, 3))
    @threads for i in 1:size(X, 3)
        println("coefficients for stripe $i")
        @views svd_coefficients!(C[:, :, i], X[:, :, i])
    end
    return C
end
function svd_coefficients!(C::AbstractMatrix, X::AbstractMatrix)
    U, S, V = svd(X)
    n = size(C, 1)
    Un = @view U[:, 1:n]
    mul!(C, Un', X) # projection of X onto reduced orthonormal basis
end
# svd coefficients but based on relative tolerance, instead of number of singular values
# returns vector of vectors, i.e. coefficients for for each position
function svd_coefficients(Y::AbstractMatrix, rtol::Real = 1e-2)
    U, S, V = svd(Y)
    tol = maximum(norm, eachcol(Y)) * rtol
    i = findlast(≥(tol), S)
    Vt = @view V'[1:i, :]
    @. Vt = S[1:i]
    y = [v for v in eachrow(Vt)] # each element is vector of singular coefficients by position
end
#############################
# generalized coefficient of determination, where errors are weighted by their standard deviation
# McFadden (in the context of logistic regression), (log?) likelihood ratio index
# x is data, y is model prediction, σ is expected noise for each data point
function likelihood_ratio(x::AbstractVector, y::AbstractVector, σ::AbstractVector = ones(length(x)))
    δ = (x .- mean(x)) ./ σ
    tss = sum(abs2, δ)
    δ = (x .- y) ./ σ
    ess = sum(abs2, δ)
    r2 = 1 - ess / tss
end

# interpolating xrd patterns on common q-grid
function interpolate_xrd_stripe(x::AbstractVector, Y::AbstractMatrix, q::AbstractVector)
    dx = diff(x)
    all(x->isapprox(x, dx[1], atol = 1e-5), dx) || throw("q not uniformly spaced")
    x = range(extrema(x)..., length = length(x))
    Yq = similar(Y)
    for i in 1:size(Y, 2)
        y = @views Y[:, i]
        f = CubicSplineInterpolation(x, y)
        @. Yq[:, i] = f(q)
    end
    return Yq
end
