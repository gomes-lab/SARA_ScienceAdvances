using HDF5
using JSON
savepath = "SARA/NatCom2020/outer/data/"
xrd_file = "Bi2O3_19F44_01_outer_xrd_gradients_input_noise_iSARA.h5"

# savefile = "Bi2O3_19F44_01_outer_xrd_gradients_no_input_noise.h5"
xrd_f = h5open(savepath * xrd_file, "r")
xrd_temperatures = read(xrd_f, "temperatures")
xrd_dwelltimes = read(xrd_f, "dwelltimes")
xrd_coefficients = read(xrd_f, "coefficients")
close(xrd_f)

xrd_f = h5open(savepath * "Bi2O3_19F44_01_outer_xrd_data.h5", "r")
xrd_positions = read(xrd_f, "positions")
close(xrd_f)

include("../inner/inner_load_data.jl")
path = "/Users/sebastianament/Documents/SEA/XRD Analysis/SARA/Bi2O3_19F44_01/"
# path = ".."
file = "Bi2O3_19F44_01_inner_loop_data.json"
f = JSON.parsefile(path * file)
optical_positions, optical_coefficients, rescaling_parameters, optical_temperatures, optical_dwelltimes = load_data(f)

using Plots
plotly()
for i in 1:8
    p = 10
    match = (T, τ) -> xrd_temperatures[end, i] ≈ T && xrd_dwelltimes[end, i] ≈ τ
    j = 1
    while true
        if match(optical_temperatures[j]-p, optical_dwelltimes[j])
            break
        end
        j += 1
    end
    # j = findfirst(match, [optical_temperatures, optical_dwelltimes])
    plot(yscale = :log10)
    # xrd = xrd_coefficients[1, :, i]
    # opt = optical_coefficients[j][1]

    # xrd = abs.(diff(xrd))
    # opt = abs.(diff(opt))
    xrd = sum(abs2, diff(xrd_coefficients[:, :, i], dims = 2), dims = 1)
    opt = [optical_coefficients[j][k1][k2] for k1 in 1:16, k2 in 1:151]
    opt = sum(abs2, diff(opt, dims = 2), dims = 1)
    xrd = vec(xrd)
    opt = vec(opt)
    xrd ./= maximum(xrd)
    opt ./= maximum(opt)
    xrd_pos = xrd_positions[:, i]
    opt_pos = optical_positions[j]
    xrd_pos = xrd_pos[1:end-1]
    opt_pos = opt_pos[1:end-1]
    plot!(xrd_pos, xrd, label = "xrd")
    plot!(opt_pos, opt, label = "opt")
    gui()
end
