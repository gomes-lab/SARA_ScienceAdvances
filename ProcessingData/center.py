"""
Compute center from images
"""
from scipy import ndimage
from scipy import signal
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

