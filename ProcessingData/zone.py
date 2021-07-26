#!/usr/bin/env python3
import sys
import PIL.Image as Image
import os
import center as GI
import data_storage as ds
import numpy as np
import json
#from Wafer import *


class zone():
    """
    Class to get spectra, either through a filename or through sockets
    """
    def __init__(self):
        """
        Initilize
        """
        self.Tpeak  = None
        self.dwell  = None
        self.pos    = None   #Position of the stripe 
        self.spec_pos = None #Position of the last spectum taken
        self.img_pos  = None #Position of the last image taken
        self.spectrometer_dict = None
        self.wl = None
        self.reference_data = None #Mirror and blank information
        self.WorkDir = ""
        self.img_call = 0
        self.spec_call = 0
        self.img_fn = []
        self.spec_fn = []
        self.spectra_zone = []

    def spectra_from_file(self, fn_data, fn_mirror, fn_blank):
        """
        Read spectra from file and normalize
        """
        print(fn_data)
        spec = ds.spec()
        data =   spec.read_spec(fn_data)
        mirror = spec.read_mirror_blank(fn_mirror)
        blank =  spec.read_mirror_blank(fn_blank)
        normal_data = spec.normalize(data["Spec"], mirror["Spec"], blank["Spec"])
        meta = data["Meta"]
        meta["delta start"] = meta["Position"]["Range"][0]
        meta["delta end"] = meta["Position"]["Range"][1]
        wl = data["Wavelengths"]
        self.Tpeak = meta["Tpeak"]
        self.dwell = meta["dwell"]
        self.pos   =   meta["Position"]["Center"]
        self.spec_pos =   meta["Position"]["Center"]
        self.spec_xdomain = [self.pos[0] + meta["delta start"][0], self.pos[0] + meta["delta end"][0]]
        self.spec_n = meta["Position"]["Scan lines"]
        return wl, normal_data, mirror, blank, meta

    def image_from_file(self, fn_image, img_spec_offset):
        """
        Read image from file
        """
        print(fn_image)
        img = Image.open(fn_image)
        #Get the optical image analysis
        img_center, img, img_info, img_data = GI.get_center(camera_client = None, camera_info = None, img = img, img_info = None)
        img_peaks = []
        scale = img_spec_offset.scale
        for d in img_data:
            if "TR_idx" in d.keys():
                for p in d["TR_idx"]:
                    img_peaks.append(p)
        width, height = img.size
        self.img_pos = self.pos
        self.img_xdomain = [self.pos[0] - 0.5 * width * img_spec_offset.scale,  self.pos[0] + 0.5 * width * img_spec_offset.scale]
        self.img_ydomain = [self.pos[1] - 0.5 * height * img_spec_offset.scale, self.pos[1] + 0.5 * height * img_spec_offset.scale]
        self.img_width = width
        self.img_height = height
        self.img_center = self.img_xdomain[0] + img_center/self.img_width * (self.img_xdomain[1] - self.img_xdomain[0])
        img_peaks = self.img_xdomain[0] + np.array(img_peaks)/self.img_width * (self.img_xdomain[1] - self.img_xdomain[0])
        return img, img_center, img_info, img_data, img_peaks.tolist()

