################# load data
using JSON
using HDF5
function load_data(f)
    nstripes = length(f)
    coeffs_1 = f[1]["legendre_coeffs"]
    num_pos = length(coeffs_1)
    num_coeffs = length(coeffs_1[1])

    log10_τ = zeros(nstripes)
    position = zeros(num_pos, nstripes)
    # optical coefficients: coefficients x positions on stripe x stripe id
    optical = zeros(num_pos, num_coeffs, nstripes)
    helper(x) = tuple(x...)
    rescaling_parameters = []
    T_max = zeros(nstripes)
    log10_τ = zeros(nstripes)
    for i in 1:nstripes
        # stripe conditions
        T_max[i] = f[i]["meta"]["temp"]
        log10_τ[i] = log10(f[i]["meta"]["dwell"])

        # centered position on the wafer
        x = float.(f[i]["pos_wafer"]) # position
        c = float(f[i]["center_real"]) # center
        @. position[:, i] = x - c

        # optical coefficients
        coeffs_i = f[i]["legendre_coeffs"]
        for j = 1:num_pos
            optical[j, :, i] = float.(coeffs_i[j])
        end

        # rescaling parameters
        push!(rescaling_parameters, helper.(f[i]["rescaling_parameters"]))
    end
    position = [x for x in eachcol(position)] # convert to vector for each stripe index
    optical = [[x for x in eachcol(optical[:, :, i])] for i in 1:size(optical, 3)]
    return position, optical, rescaling_parameters, T_max, log10_τ
end
