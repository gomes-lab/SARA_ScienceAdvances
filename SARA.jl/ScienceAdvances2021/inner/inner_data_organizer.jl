# takes in both legendre coefficients and rescaling,
# matches meta-data and writes to single file
using SARA
using JSON
using HDF5

# returns true if dwell time and peak temperature are the same
same_meta(f) = g->same_meta(f, g)
function same_meta(f, g)
    fm = f["meta"]
    gm = g["meta_data"]
    fm["temp"] == gm["Tpeak"] && fm["dwell"] == gm["dwell"]
end

# put path to legendre_coefficients.json and bias.json here
path = "/Users/sebastianament/Documents/SEA/XRD Analysis/SARA/Bi2O3_19F44_01/"
file = "legendre_coefficients.json"
f = JSON.parsefile(path * file)
f = f["spectra"] # don't need wavelengths here

file = "bias.json"
g = JSON.parsefile(path * file)

const broken_index = 210 # not usable data
deleteat!(f, broken_index)
indices = zeros(Int, length(f))
for i in 1:length(f)
    fi = f[i]
    j = findfirst(same_meta(fi), g)
    if isnothing(j) # used to find broken index
        println(i)
        continue
    end
    indices[i] = j
    fi["rescaling_parameters"] = g[j]["rescaling_parameters"]
    f[i] = fi # technically not necessary since dictionaries are references
end

h = open(path * "Bi2O3_19F44_01_inner_loop_data.json", "w")
JSON.print(h, f)
close(h)
