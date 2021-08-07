# preparing XRD data for gradient analysis
using Base.Threads

# 1. load data
using HDF5
path = "/Users/sebastianament/Documents/SEA/XRD Analysis/SARA/Bi2O3_19F44_01/"
file = "Bi2O3_19F44_01_all_oned.h5"
datafile = h5open(path * file);
data = datafile["exp"];

stripenames = keys(data);
nstripes = length(stripenames)

temperatures = zeros(nstripes)
dwelltimes = zeros(nstripes)

nspec = length(names(data[stripenames[1]]))
nQ = size(read(data[stripenames[1]]["1"], "integrated_1d"), 1)
X = zeros(nQ, nspec, nstripes)
Y = similar(X)
for j in eachindex(stripenames)
    println("getting data for stripe $j")
    stripe = data[stripenames[j]]
    conditions = split(stripenames[j], "_")
    temperatures[j] = parse(Float64, conditions[4])
    dwelltimes[j] = parse(Float64, conditions[2])
    for i in 1:nspec
        d = stripe["$(i-1)"]
        XY = read(d, "integrated_1d")
        X[:, i, j], Y[:, i, j] = eachcol(XY)
    end
end

# take log10 of dwelltime
@. dwelltimes = log10(dwelltimes)

# 3. get centered position
using JSON
function get_positions(path, nspec, stripenames)
    nstripes = length(stripenames)
    posfile = JSON.parsefile(path * "xrd_centers.json")
    step = 10 # the positions are 10 μm appart
    pos = range(0, length = nspec, step = step) # μm
    pos = collect(pos) ./ 1e3 # mm
    positions = zeros(nspec, nstripes)
    for i in 1:nstripes
        ci = posfile[stripenames[i]] + 1 # add one because python indexing
        ci = convert(Int, ci)
        @. positions[:, i] = pos .- pos[ci]
    end
    return positions
end
positions = get_positions(path, nspec, stripenames)

# 4. interpolate onto same q grid
using Interpolations
function interpolate_xrd(X::AbstractArray, Y::AbstractArray)
    # q = range(extrema(Q)..., length = nQ)
    # q = range(10, 40, length = nQ) # cut off noisy parts
    p = .03 # cut off percentage of Q domain
    minQ = maximum(minimum.(eachcol(X[:, 1, :])))
    maxQ = minimum(maximum.(eachcol(X[:, 1, :])))
    q = range(minQ * (1+p), maxQ * (1-p), length = size(X, 1)) # interpolation grid
    Yq = zeros(length(q), size(Y, 2), size(Y, 3))
    @threads for j in 1:size(Y, 3)
        println("interpolating stripe $j")
        for i in 1:size(Y, 2)
            x, y = @views X[:, i, j], Y[:, i, j]
            dx = diff(x)
            all(x->isapprox(x, dx[1], atol = 1e-5), dx) || throw("q not uniformly spaced")
            x = range(extrema(x)..., length = length(x))
            f = CubicSplineInterpolation(x, y)
            @. Yq[:, i, j] = f.(q)
        end
    end
    return q, Yq
end

q, spectrograms = interpolate_xrd(X, Y)

# 5. exclude bad xrd stripes
using DelimitedFiles
function good_indices(data)
    i = string.(@view readdlm(path * "exclude_xrd.txt", '.')[:, 1])
    findall(!in(i), names(data))
end
ind = good_indices(data)

spectrograms = spectrograms[:, :, ind]
temperatures = temperatures[ind]
dwelltimes = dwelltimes[ind]
positions = positions[:, ind]

path = "SARA/NatCom2020/outer/data/"
f = h5open(path * "Bi2O3_19F44_01_outer_xrd_data.h5", "w")
f["q"] = collect(q)
f["spectrograms"] = spectrograms
f["dimensions"] = ["q", "position", "stripe"]
f["positions"] = positions
f["temperatures"] = temperatures
f["dwelltimes"] = dwelltimes
close(f)
close(datafile)

# path = "SARA/NatCom2020/outer/data/"
# f = h5open(path * "Bi2O3_19F44_01_outer_xrd_data.h5", "r")
# q = read(f, "q")
# spectrograms = read(f, "spectrograms")
# positions = read(f, "positions")
# temperatures = read(f, "temperatures")
# dwelltimes = read(f, "dwelltimes")
# close(f)
#
# using Plots
# plotly()
# i = 2
# heatmap(positions[:, i], q, @views log10.(spectrograms[:, :, i]))
# plot(positions[:, i], coefficients[:, :, i]')
# scatter(positions[:, i], coefficients[2, :, i])
# gui()
# temperatures[i]
# dwelltimes[i]
