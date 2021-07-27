"""
Compute center from images
"""
from scipy import ndimage
from scipy import signal
from scipy.spatial import distance
import sys
from PIL import Image
from transition_finder import *
import copy as cp
import time

def get_center(camera_client = None, camera_info = None, img = None, img_info = None, img_spec_offset = None):
    """
    One-shot center determination by getting an image and sending it to the
    image processor
    """
    msg_id = 101
    if camera_info is None and camera_client is not None:
        camera_info = camera_client.get_DCX_GET_CAMERA_INFO(msg_id)
        ndim = get_color_dim(camera_info)
    if img is not None:
        ndim = len(img.getbands())
    if img is None:
        img, img_info = camera_client.get_image(msg_id)
    img_arr = np.array(img)
    if ndim == 1:
        img_arr = np.expand_dims(img_arr, axis = 2)
    img_data = []

    #Analyze image using D.S. image processing
    for c in range(ndim):
        d = HSI_Tr_finder2(img_arr[:,:,c], mpath=None, blur=5, Gpromfilt=100,
                c_thresh=0.35, v_cutoff=0, h_thresh=3.0, plotting=False, norm=False, ImgIO=True, pix_siz = 1.0)
        img_data.append(d)
    #The centers along the different color channels are weighted according to their S/N ratio
    weights = []
    centers = []
    for d in img_data[:] :
        weights.append(d['S/N']**2)
        centers.append(d['center'])
    center = np.average(centers, weights = weights)
    return center, img, img_info, img_data

#Find LSA center
def get_center_cos_corr(data):
#Using the cosine method
    maximum = 1.
    for i in range(int(round((data.shape[1])*0.35)),int(round((data.shape[1])*0.65))):
        delta = min(i,data.shape[1]-i)
        dl = data[:,i-delta:i]
        dr = data[:,i:i+delta]
        norm = 0.
        for j in range(dl.shape[1]):
            norm += distance.cosine(dl[:,-j] , dr[:,j])
        norm = norm/dl.shape[1]
        if norm < maximum:
            maximum = norm
            imaximum = i
#Using the correlation method
    im1 = np.array(data[:,:])
    im = cp.deepcopy(im1)
    center = []
    weights = []
    weights_max = []
    corr = []
    for i in range(im1.shape[0]):
        result = ndimage.correlate(im1[i,:],np.flip(im1[i,:],0), mode='wrap')
        result = np.array(result)-np.mean(np.array(result))
        result = fft_smoothing(np.array(result),25.)
        im[i,:] = result
        center.append(np.argmax(result))
        #print(np.argmax(result),np.amax(result)**2)
        corr.append(result)
        weights.append(sum(np.abs(im1[i,:])))
        weights_max.append(np.amax(result)**4)
    corr = np.array(corr)
    smooth = []
    for i in range(corr.shape[1]):
        smooth.append(np.mean(corr[:,i]))
    smd = fft_smoothing(np.array(smooth),25.)
    center = np.array(center)
    weights = np.array(weights)
    imaximum_conv = (len(result))*0.5 + (np.average(center,weights=weights_max)-(len(result))*0.5)*0.5
    return imaximum, imaximum_conv

#Helper functions
def fft_smoothing(d,param):
#d is a 1D vector, param is the cutoff frequency
    rft = np.fft.rfft(d)
    rft[int(param):] = 0.
    d = np.fft.irfft(rft,len(d))
    return d
