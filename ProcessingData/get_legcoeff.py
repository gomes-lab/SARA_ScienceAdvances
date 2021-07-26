#!/usr/bin/env python3
'''
Script to analyze spectroscopy data. 
- Normalizes the reflectance and stores the resulting maps as png files
- Extracts the Legendre coefficients and stores them in legendre_coefficients.json
'''
import json
import seaborn as sns
import itertools
import copy as cp
import numpy as np
import math as mt
import glob
import matplotlib.pyplot as plt
from numpy.polynomial import polynomial
from matplotlib import cm
from numpy.polynomial import legendre
import zone as LSA_Zone

def get_leg_fit1d(data,xval,deg):
    x = (xval-np.min(xval)) ; x = x/np.max(x)*2.-1.
    pfit, meta_data = np.polynomial.legendre.legfit(x,data,deg,full = True)
    residual = meta_data[0]
    return pfit,residual

def eval_leg_fit1d(xval,m):
    x = (xval-np.min(xval)) ; x = x/np.max(x)*2.-1.
    return legendre.legval(x, m)

#Source data
fn_mirror = 'Bi2O3/Reflectance/mirror_00.csv' 
fn_blank =  'Bi2O3/Reflectance/blank_00.csv' 
data_files = glob.glob('Bi2O3/Reflectance/s*_*_*.csv')

#Fit functions, cutting off wave lengths below lmin and above lmax (index)
lmin = 400
lmax = 1600

dataset_all = {}
dataset_spectra = []
for fn_data in data_files[:]:
    #Read data
    zone = LSA_Zone.zone()
    wl, normal_data, mirror, blank, meta = zone.spectra_from_file(fn_data, fn_mirror, fn_blank)
    dump_data = {}
    dump_data["filename"] = fn_data

    normal_data = normal_data[lmin:lmax,:]
    dump_data["meta"] = meta

    #Number of coeffs to fit
    cdeg = 16

    #Coefficient fitting dimenison
    x = np.linspace(-1,1,num=normal_data.shape[0],endpoint=True) #Dimension along wavelength

    #Position on wafer in mm
    #pos_wafer = np.linspace(meta['delta start'][0] + meta['scan center'][0],meta['delta end'][0] + meta['scan center'][0],normal_data.shape[1])
    pos_wafer = np.linspace(meta['Position']['Range'][0][0] + meta['Position']['Center'][0] , meta['Position']['Range'][1][0] + meta['Position']['Center'][0], normal_data.shape[1])

    #Fit every stripe
    cfit = []
    for i in range(normal_data.shape[1]):
        m, residual = get_leg_fit1d(normal_data[:,i], x, cdeg)
        cfit.append(m.tolist())
    dump_data["legendre_coeffs"] = cfit
    cfit = np.array(cfit)

    #Plot raw data
    title_str = "Dwell "+str(meta["dwell"])+"\u03bcs, Tpeak "+str(meta["Tpeak"])+"℃"
    plt.imshow(np.array(normal_data),extent=[min(pos_wafer),max(pos_wafer),min(wl),max(wl)], aspect='auto')
    plt.title(title_str)
    plt.xlabel('Position (mm)')
    plt.ylabel('Wavelength (nm)')
    plt_out = fn_data.replace("csv", "png").replace("s", "raw")
    plt.savefig(plt_out, format='png')
    plt.close()

    #Plot fitted data
    normal_data_recon=[]
    for i_pos in range(len(pos_wafer)):
        normal_data_recon.append(eval_leg_fit1d(x, cfit[i_pos, :]))
    plt.imshow(np.array(normal_data_recon).T,extent=[min(pos_wafer),max(pos_wafer),min(wl),max(wl)], aspect='auto')
    plt.title(title_str)
    plt.xlabel('Position (mm)')
    plt.ylabel('Wavelength (nm)')
    plt_out = fn_data.replace("csv", "png").replace("s", "reconstructed")
    plt.savefig(plt_out, format='png')
    plt.close()


    #Number of dimensions to sample (i.e., the higherst Legendre coefficient) to learn
    dump_data["pos_wafer"] = pos_wafer.tolist()
    dataset_spectra.append(dump_data)


    #Store metadata
    logtau = np.log10(meta["dwell"])
    logtau_ms = np.log10(meta["dwell"]*0.001)
    dump_data["logtau"] = logtau.tolist()
    dump_data["logtau_ms"] = logtau_ms.tolist()
    Tpeak = meta["Tpeak"]

dataset_all["wavelengths"] = wl[lmin:lmax].tolist()
dataset_all["spectra"] = dataset_spectra
fn_dump = "legendre_coefficients.json"
dump_file = open(fn_dump, 'w')
json.dump(dataset_all, dump_file, sort_keys=False, indent=2)
dump_file.close()
