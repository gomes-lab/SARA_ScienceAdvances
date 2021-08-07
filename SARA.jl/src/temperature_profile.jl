struct TemperatureProfile{AMP, FWHM, U, V}
    # amplitude and full-width half-max functions for generalized Gaussian temperature profile
    amplitude::AMP
    width::FWHM
    # parameters governing uncertainty calculation
    dT::U
    dx::V
end

# Constructors:
# defaults to parameters of Science Advances submission
function TemperatureProfile(dT::Real = 20, dx::Real = 10/1000)
    get_temperature_profile_Bi2O3_2021(dT, dx)
end

# template for t-profile constructor
function get_temperature_profile_template(dT::Real = 20, dx::Real = 10/1000)
    println("temperature_profile_template")
    amplitude(T_max, log10_τ) = 2
    function width(T, log10_τ)
        throw("not implemented - should be a functional expression from temperature and dwell time to the fwhm of the profile")
    end
    TemperatureProfile(amplitude, width, dT, dx)
end

# temperature profile for the Bi2O3 data in the Science Advances submission
function get_temperature_profile_Bi2O3_2021(dT::Real = 20, dx::Real = 10/1000)
    amplitude(T_max, log10_τ) = 2
    function width(T, log10_τ)
        fwhm_0 = 1.60742002e+03
        a = (-1.46900821e+00, -2.06060111e+02, 3.05703021e-01, 3.29193895e+02, -1.23361265e+13, 1.09165256e+01, -1.69547934e+04, -1.42207444e-04)
        fwhm_0 + a[1]*T + a[2]*log10_τ + a[3]*log10_τ*T + a[4]*log10_τ^4/T +
            a[5]/log10_τ^8/T^3 + a[6]*T/log10_τ^3 + a[7]/log10_τ^4 + a[8]*T^3/log10_τ^8
    end
    TemperatureProfile(amplitude, width, dT, dx)
end

# OLD temperature profile for the Bi2O3 data
function get_temperature_profile_Bi2O3_2020(dT::Real = 20, dx::Real = 10/1000)
    amplitude(T_max, log10_τ) = 2
    function width(T, log10_τ)
        fwhm_0 = 6.21599199e+01
        a = (-3.56990024e-01, 2.84613876e+02, -8.90507983e-03, -5.76367089e+02, -1.95522786e+13, 5.51350589e+00, -3.34684841e+03, -1.33972469e-04)
        fwhm_0 + a[1]*T + a[2]*log10_τ + a[3]*log10_τ*T + a[4]*log10_τ^4/T +
            a[5]/log10_τ^8/T^3 + a[6]*T/log10_τ^3 + a[7]/log10_τ^4 + a[8]*T^3/log10_τ^8
    end
    TemperatureProfile(amplitude, width, dT, dx)
end

# temperature profile for CHESS 2021
function get_temperature_profile_CHESS_2021(dT::Real = 20, dx::Real = 10/1000)
    amplitude(T_max, log10_τ) = 2
    # power = current in A
    # tau = dwell in us
    function _std(power, log10_τ)
        mm_per_pixel = 0.00153846153 # millimeters per pixel
        μm_per_pixel = mm_per_pixel * 1e3 # micrometers per pixel
        μm_per_pixel * (-6.0585 * power + 5.29285 * log10_τ + 1.432e-2 * power^2 - 1.59943e1 *log10_τ^2 + 7.6715e-1 * power*log10_τ + 726.0242)
    end
    _power(T_max::Int, log10_τ) = _power(float(T_max), float(log10_τ))
    _power(T_max, log10_τ) = (5000*sqrt(5)*sqrt(614000*T_max+1327591125*log10_τ^2 -8080505000*log10_τ+12142813538)-341554591*log10_τ+1065621710)/3070000
    _fwhm(power, log10_τ) = _std(power, log10_τ) / 2 # by the definitions of variables in the Gaussian-like function Max used
    width(T_peak, log10_τ) = _fwhm(_power(T_peak, log10_τ), log10_τ) # getting the power from the power profile
    TemperatureProfile(amplitude, width, dT, dx)
end

# Functor definitions:
# WARNING: temperature profile is a function of position in mm
function (P::TemperatureProfile)(T_max::Real, log10_τ::Real)
    a = P.amplitude(T_max, log10_τ)
    f = P.width(T_max, log10_τ)
    x->_T_helper(T_max, f, a, 1e3*x) # multiplying position by 1e3 to convert from mm to μm
end
function (P::TemperatureProfile)(T_max::Real, log10_τ::Real, x::Real)
    P(T_max, log10_τ)(x)
end
# WARNING: temperature profile is a function of position in μm
function _T_helper(T_max, fwhm, ampl, x::Real)
    return @. T_max * exp(-abs(2x / fwhm)^ampl)
end

profile(P::TemperatureProfile, T_max, log10_τ) = P(T_max, log10_τ)
# returns the positive side of the inverse mapping T to position in mm
function inverse_profile(P::TemperatureProfile, T_max::Real, log10_τ::Real)
    a = P.amplitude(T_max, log10_τ)
    f = P.width(T_max, log10_τ)
    function invprof(T)
        T > 0 || throw(DomainError("T is non-positive: $T"))
        T_max ≥ T || throw(DomainError("T exceeds T_max: T = $T > $T_max = T_max"))
        (T ≈ T_max) ? zero(eltype(T)) : log(T_max / T)^(1/a) * f/2 * (1e-3)
    end
end

# x is a position
# computes uncertainty of the temperature profile
# Two components:
# - multiplicative and
# - proportional to derivative
function temperature_uncertainty(P::TemperatureProfile, T_max, log10_τ, dT::Real = P.dT, dx::Real = P.dx)
    f = P(T_max, log10_τ)
    function temperature_variance(x::Real)
        T, dTdx = f(x), ForwardDiff.derivative(f, x)
        (dT * T / 1400)^2 + (dx * dTdx)^2 # error variance is additive
    end
end
