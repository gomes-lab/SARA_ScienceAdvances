using HDF5
using SARA: TemperatureProfile
savepath = "SARA/NatCom2020/outer/data/"
xrd_file = "Bi2O3_19F44_01_outer_xrd_gradients.h5"
# xrd_file = "Bi2O3_19F44_01_outer_xrd_gradients_no_input_noise.h5"

# loading xrd data
xrd_f = h5open(savepath * xrd_file, "r")
xrd_temperatures = read(xrd_f, "temperatures")
xrd_dwelltimes = read(xrd_f, "dwelltimes")
xrd_gradients = read(xrd_f, "gradients")
xrd_var_gradients = read(xrd_f, "var(gradients)")
xrd_coefficients = read(xrd_f, "coefficients")
close(xrd_f)

xrd_f = h5open(savepath * "Bi2O3_19F44_01_outer_xrd_data.h5", "r")
xrd_positions = read(xrd_f, "positions")
close(xrd_f)

####### loading optical data
optical_file = "Bi2O3_19F44_01_outer_optical_gradients.h5"
optical_f = h5open(savepath * optical_file, "r")
optical_temperatures = read(optical_f, "temperatures")
optical_dwelltimes = read(optical_f, "dwelltimes")
optical_gradients = read(optical_f, "gradients")
optical_var_gradients = read(optical_f, "var(gradients)")
close(optical_f)

include("../inner/inner_load_data.jl")
path = "/Users/sebastianament/Documents/SEA/XRD Analysis/SARA/Bi2O3_19F44_01/"
# path = ".."
file = "Bi2O3_19F44_01_inner_loop_data.json"
f = JSON.parsefile(path * file)
optical_positions, optical_coefficients, optical_rescaling_parameters,
        optical_inner_temperatures, optical_inner_dwelltimes = load_data(f)

using Plots
plotly()
dT = 50 # uncertainty in peak temperature in C
dx = 100/1000 # uncertainty in position in mm
P = TemperatureProfile(dT, dx)
function get_match(T_max, log10_τ, temperatures, dwelltimes)
   j = 1
   while true
      if T_max == temperatures[j] && log10_τ == dwelltimes[j]
          break
      end
      j += 1
   end
   return j
end

using SARA: iSARA, get_temperature_process
using GaussianDistributions
for i in 3:3
    T_max = xrd_temperatures[end, i] + 10 # adding 10 because of offset from T peak for outer temperature
    log10_τ = xrd_dwelltimes[end, i]
    j = get_match(T_max, log10_τ, optical_temperatures[end, :].+10, optical_dwelltimes[end, :])
    println("T_max = $T_max, $(optical_temperatures[end, j]+10), $(optical_inner_temperatures[j])")
    println("log10_τ = $log10_τ, $(optical_dwelltimes[end, j]), $(optical_inner_dwelltimes[j])")

    σ = .01
    l = .1
    θ = optical_rescaling_parameters[j]
    k = iSARA(l, θ)

    # coefficients as a function of position
    ncoeff = 3
    coeff_ind = 1:3 #[1, 2, 3, 4] #8, 16] # coefficient indices
    p2 = plot(layout = grid(1, ncoeff)) #@layout [a b c d]) # xlabel = "position (mm)", title = "coefficients"

    # GP for xrd data
    xrd_coeff = xrd_coefficients[coeff_ind, :, i]
    xrd_pos = xrd_positions[:, i]
    c = [conditional(Gaussian(k), xrd_pos, xrd_coeff[ci, :], σ^2) for ci in coeff_ind]
    cx = [mean(ci)(xi) for ci in c, xi in xrd_pos]
    plot!(xrd_pos, xrd_coeff', label = "xrd")
    plot!(xrd_pos, cx', label = "xrd gp")

    # GP for optical data
    opt_coeff = [optical_coefficients[j][ci][cj] for ci in coeff_ind, cj in 1:151]
    opt_pos = optical_positions[j]
    # cent, fom = center_of_symmetry(opt_coeff)
    # @. opt_pos -= opt_pos[cent]
    c = [conditional(Gaussian(k), opt_pos, opt_coeff[ci, :], σ^2) for ci in coeff_ind]
    cx = [mean(ci)(xi) for ci in c, xi in opt_pos]
    plot!(opt_pos, opt_coeff', label = "optical")
    plot!(opt_pos, cx', label = "opt gp")

    # coefficients as a function of temperature
    T_profile = P(T_max, log10_τ)

    xrd_temp = T_profile.(xrd_positions[:, i])
    opt_temp = T_profile.(optical_positions[j])
    p3 = plot(layout = grid(1, ncoeff)) # xlabel = "position (mm)", title = "coefficients"

    # xrd
    input_noise = Val(true)
    c = [get_temperature_process(Gaussian(k), xrd_pos, xrd_coeff[ci, :], σ, P, T_max, log10_τ, input_noise) for ci in coeff_ind]
    cx = [mean(ci)(xi) for ci in c, xi in xrd_temp]
    plot!(xrd_temp, xrd_coeff', label = "xrd")
    plot!(xrd_temp, cx', label = "xrd gp")

    # optical
    c = [get_temperature_process(Gaussian(k), opt_pos, opt_coeff[ci, :], σ, P, T_max, log10_τ, input_noise) for ci in coeff_ind]
    cx = [mean(ci)(xi) for ci in c, xi in opt_temp]
    plot!(opt_temp, opt_coeff', label = "optical")
    plot!(opt_temp, cx', label = "opt gp")

    p1 = plot(xlabel = "temperature (C)", title = "GP gradients", yaxis = :log10)
    plot!(xrd_temperatures[:, i], xrd_gradients[:, i],
            #ribbon = 2sqrt.(xrd_var_gradients[:, i]),
            label = "xrd")
    plot!(optical_temperatures[:, j], optical_gradients[:, j],
            #ribbon = 2sqrt.(optical_var_gradients[:, j]),
            label = "optical")

    local l = @layout [a; b; c]
    plot(p2, p3, p1, layout = l, legend = false) #, title = "T_max = $T_max, log10_τ = $log10_τ")
    gui()
end
