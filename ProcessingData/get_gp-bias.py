#!/usr/bin/env python3
"""
Script to extract the gp bias features from microscopy images
"""
import sys
import json
import os
import copy as cp
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib
from numpy.polynomial import polynomial
import offsets as GS
from probability_dist import *
import data_storage as ds
import zone as LSA_Zone
from os import listdir
from matplotlib import cm
from collections import OrderedDict
import seaborn as sns
import itertools

#Set color schemes
cmaps = OrderedDict()
cmaps['Qualitative'] = ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']
plt.rcParams["image.cmap"] = "Set1"
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set1.colors)
palette = itertools.cycle(sns.color_palette("muted"))
palette = sns.color_palette("muted")

def list_files(directory, extension):
    return [f for f in listdir(directory) if f.endswith('.' + extension)]

def convert_bias_parameters(bias_parameters, center):
    """
    Converts the sum of Gaussian parameters into a format the the GP can
    interpret
    """
    bias_parameters_new = []
    for b in bias_parameters:
        std = b[2] * 0.5 * np.sqrt(2.*np.log(2.))
        bb = (b[0] * b[4], b[1] - center, std, b[3])
        bias_parameters_new.append(bb)
    return bias_parameters_new

def get_img_filename(pos, image_error, bx = 1., by = 1.):
    """
    Convert position to a filename
    """ 
    lsa = ds.LSA()
    stripe = {}
    stripe["x"] = pos[0]
    stripe["y"] = pos[1]
    if image_error:
        stripe["x"] = round(pos[0]/bx)
        stripe["y"] = round(pos[1]/by)
    stripe["dwell"] = 0. 
    stripe["Tpeak"] = 0. 
    fn = lsa.image_name(stripe)
    fn = fn[:9]
    return fn

def get_filename(pos, img_dir, bx = 1., by = 1.):
    """
    Captures an image with given settings.
    """
    fn = get_img_filename(pos, image_error = True, bx = bx, by = by)
    fn += "*.bmp"
    if 'img_dir' in locals():
        fn = os.path.join(img_dir, fn)
    img_fn = glob.glob(fn)
    if len(img_fn) > 0:
        img_fn = sorted(img_fn)[0]
        img = Image.open(img_fn)
        mode = img.mode
        if mode == "RGB":
            r, g, b = img.split()
            img = Image.merge("RGB", (b, g, r))
    return img, img_fn

rescaling_datas = []

img_dir = "Bi2O3/Images/"
files = list_files(img_dir, "bmp")
exclude = []
for f in files[:]:
    if f in exclude:
        continue
    rescaling_data = {}
    
    #Parse information from the filename
    meta_img = {}
    fn_meta = f.split("_")

    #The last part is the temperature in C
    meta_img["Tpeak"] = float(fn_meta[-1].split(".")[0])

    #The second last part is the temperature in dwell time in microsec
    meta_img["dwell"] = float(fn_meta[-2])
    meta_img["logtau"] = np.log10(float(fn_meta[-2]))
    meta_img["pos"] = [float(fn_meta[0][1:])*2, float(fn_meta[1])*5]
    meta_img["filename"] = f
    pos = meta_img["pos"]
    
    img, img_fn = get_filename(pos, img_dir, bx = 2., by = 5.)
    plt_out = img_fn.replace("bmp", "png").replace("b", "aa")
    
    zone = LSA_Zone.zone()
    img_spec_offset = GS.img_spec_offset()
    img_spec_offset.scale = 0.00092             #Scaling of pixels in mm
    img_spec_offset.scale_imgcam = 0.0006680932 #Scaling of pixels in mm for imaging camera
    img_spec_offset.offset  = 0                 #Offset of the spectrometer with respect to the image center in pixels.
    img_spec_offset.offsety = 0                 #Offset of the spectrometer with respect to the image center in pixels.
    img_spec_offset.img_shift = img_spec_offset.offset * img_spec_offset.scale    #The amount of shift along the x-axis in mm of the spectrum with respect to image
    img_spec_offset.offset_global = [0., 0.]
    
    zone.pos = pos
    pd = probability_dist()
    img, img_center_px, img_info, img_data, img_peaks = zone.image_from_file(img_fn, img_spec_offset)
    if abs(img_center_px - zone.img_width * 0.5) > zone.img_width*0.1:
        img_center_px = 0.5 * zone.img_width 
    img_center = zone.img_xdomain[0] + img_center_px/zone.img_width * (zone.img_xdomain[1] - zone.img_xdomain[0])
    spec_center = img_center
    peaks = np.array(img_peaks)
    n_dense = 800
    zone.spec_xdomain = [img_center-1.75, img_center+1.75]
    x_plot = np.linspace(zone.spec_xdomain[0], zone.spec_xdomain[1], n_dense).reshape(-1,1)
    dist_peaks, dist_lsa, dist_peaks_lsa, bias_parameters, LSA_width = pd.get_img_bias(peaks, img_center, spec_center, x_plot, lsa_frac = 1.)
    
    bias_parameter_centered = convert_bias_parameters(bias_parameters, img_center)
    
    #Convolve the uncertainty and the prior distribution
    dist_sum_peaks = pd.sum(dist_peaks,"SumPeaks",1.)
    dist_sum_peaks_lsa = pd.sum(dist_peaks_lsa,"SumPeaks",1.)
    
    # Plot on three seperate axes
    fig, axes = plt.subplots(nrows=2, sharex=True)
    axes = axes.tolist()
    axes[0].set_ylabel("Rescaling (a.u.)")
    axes[1].set_ylabel("y pos (mm)")
    axes[1].set_xlabel("x pos (mm)")
    
    w1 = zone.img_xdomain[0] - img_center 
    w2 = zone.img_xdomain[1] - img_center 
    h1 = zone.img_ydomain[0] - 0.5 * (zone.img_ydomain[0] + zone.img_ydomain[1])
    h2 = zone.img_ydomain[1] - 0.5 * (zone.img_ydomain[0] + zone.img_ydomain[1])
    l1, = axes[0].plot(x_plot - img_center, dist_lsa, color=palette[3], label = "LSA bias")
    axes[0].yaxis.set_ticks([])
    axes.append(axes[0].twinx())
    l2, = axes[2].plot(x_plot - img_center, dist_sum_peaks['dist'], color=palette[4], label = "RGB bias")
    axes[2].yaxis.set_ticks([])
    plt.legend([l1, l2],["LSA bias", "RGB bias"], loc = 'upper right', frameon=False)

    # Size of the image in pixels (size of orginal image) 
    width, height = img.size 
      
    # Setting the points for cropped image 
    left = 0
    top = height/2
    right = width
    bottom = height
      
    # Cropped image of above dimension 
    img = img.crop((left, top, right, bottom)) 
    width, height = img.size 

    im = axes[1].imshow(img, extent=[w1,w2,h1,h2], aspect = 'auto')
    axes[1].set_xlim([-0.55, 0.55])
    for bias_i in bias_parameter_centered[:-1]:
        axes[1].axvline(x=bias_i[1], ymin = (h2), ymax = 2.2*h2,
                color=palette[8], linewidth = 1.0)

    
    title_str = "Dwell "+str(meta_img["dwell"])+"\u03bcs, Tpeak "+str(meta_img["Tpeak"])+"℃"
    plt.title(title_str)
    plt.savefig(plt_out, format='png')
    plt.close(fig)

    
    rescaling_data["meta_data"] = meta_img
    rescaling_data["rescaling_parameters"] = bias_parameter_centered
    rescaling_datas.append(rescaling_data)

# Serializing json
json_object = json.dumps(rescaling_datas, indent = 4)

# Writing to json
with open("bias.json", "w") as outfile:
    outfile.write(json_object)
