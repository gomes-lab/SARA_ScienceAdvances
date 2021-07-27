from scipy.optimize import least_squares
import copy as cp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.special import gamma

def set_params():
    fwhm_0 = -1.16041217e+06
    m = -4.00734281e-01
    beta = 1.16164136e+06
    gamma = 1.21413457e+00
    delta =  1.50529339e+00
    ampl0 =  2.27390577e+00
    ampl1 =  -4.57034870e-04 
    ampl2 =  -1.75832969e-01
    x0 = np.array([fwhm_0,m,beta,gamma,delta,ampl0,ampl1,ampl2])
    return x0

def set_params_si():
    fwhm_0 = -7.86724363e+05
    m = -2.91013182e-01
    beta = 7.87876570e+05
    gamma = 6.62643637e-01
    delta =  2.58618404e+00
    ampl0 =  1.98348576e+00
    ampl1 =  -3.38363644e-04
    ampl2 =  -4.81178175e-02
    x0 = np.array([fwhm_0,m,beta,gamma,delta,ampl0,ampl1,ampl2])
    return x0

def T(Tpeak,FWHM,AMPL,width):
    return (Tpeak)*np.exp(-np.power(abs(2*width/FWHM),AMPL))

def ffwhm(x,T,logtau): 
    fwhm_0 = x[0]
    m      = x[1]
    beta   = x[2]
    gamma  = x[3]
    delta  = x[4]
    return fwhm_0 + m * T + beta * np.tanh(gamma*logtau+delta)

def aampl(x,T,logtau): 
    a0 = x[5]
    a1 = x[6]
    a2 = x[7]
    return a0 + a1 * T + a2 * logtau  

