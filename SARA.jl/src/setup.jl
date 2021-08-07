using SARA
using GaussianDistributions

# set up temperature profile
dT = 25 # uncertainty in peak temperature in C
dx = 50/1000 # uncertainty in position in mm
P = TemperatureProfile(dT, dx) # TODO: this needs to be updated before the run!

# define the range of temperatures that are taken from each stripe
t_low, t_high = 200, 10 # offset from T_peak
nout = 16
relevant_T = get_relevant_T(t_low, t_high, nout)

# define kernel used to calculate gradients per stripe (χAI)
l = .2 # length scale of inner GP
char_kernel = iSARA(l) # characterization kernel with length scale l

# TODO: interpolate xrd patterns

# convert stripe xrd data to global gradient data
# stripe_to_global(x, y, σ, k, P, T_max, log10_τ, relevant_T, input_noise)
# x is position
# Y is matrix of integrated xrd spectrograms as a function of position
# σ is measurement noise (can be learned using the estimate_noise function)
σ = 0.05
# T_max, log10_τ are stripe conditions
# relevant_T is function defined above
input_noise = Val(true) # switches on input noise propagation
condition = T_max, log10_τ
xrd_to_global(x, Y, σ, char_kernel, P, condition, relevant_T, input_noise)

# σAI
# stripe uncertainty acquisition function
policy = stripe_uncertainty_sampling(relevant_T)

# set up outer GP
T_length = 30.
τ_length = .3
c_length = .1 # disregard for systems without composition dimension
synth_kernel = oSARA(T_length, τ_length, c_length)
G = Gaussian(synth_kernel)

# standardize global gradient data before passing it to GP
y_normalization = std(y)
y /= y_normalization
σ² /= y_normalization^2 # noise variance

# G is GP
# x is vector / array of condition tuples
# y are corresponding gradients (vector of floats)
# variance is uncertainty in gradients (2nd return value of xrd_to_global)
# stripe_conditions is vector of potential conditions (tuples)
# policy is active learning aquisition function
next, G = next_stripe(G, x, y, variance, stripe_conditions, policy)

# evaluates GP
# function evaluate(G::Gaussian, n::Int = 256,
#                   T = range(200, 1200, length = n + 1),
#                   τ = range(2.5, 4, length = n),
#                   c = range(0, 1, length = 32))
#     return mean(G).(tuple.(T, τ', reshape(c, 1, 1, :)))
# end

# using HDF5
#
# # test workflow with LaMnOx data
# datapath = "/Users/sebastianament/Documents/SEA/XRD Analysis/SARA/LaMnOx/"
# file = "LaMnOx_outer_xrd_data.h5"
# f = h5open(datapath * file)
# positions = read(f, "positions")
# temperatures = read(f, "temperatures")
# dwelltimes = read(f, "dwelltimes")
# compositions = read(f, "compositions")
# spectrograms = read(f, "spectrograms")
# close(f)
