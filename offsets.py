"""
Compute offsets from images to spectroscopy
"""
import sys
import time
import copy as cp

class img_spec_offset():
    """
    Class containing the offsets and scaling of the image
    Must be calibrated for each camera, spectrum, and stage
    """
    def __init__(self):
        """
        Define default parameters for the LSA stage
        """
        #For the LSA stage
        #self.scale = 0.000301896855839982 #Scaling of pixels in mm
        self.scale = 0.0003048263 #Scaling of pixels in mm for CHESS
        self.scale_imgcam = 0.0006680932 #Scaling of pixels in mm for CHESS
        #self.scale = 0.0006680932 #Scaling of pixels in mm for CHESS imaging camera
        self.offset  = -59 #Offset of the spectrometer with respect to the image center in pixels. Negative, bc the camera has to move there to get the correct position
        self.offsety = 0   #Offset of the spectrometer with respect to the image center in pixels. Negative, bc the camera has to move there to get the correct position
        self.img_shift = self.offset * self.scale    #The amount of shift along the x-axis in mm of the spectrum with respect to image
        self.offset_global = [-0.420, 1.250] 

    def todict(self):
        """
        Returns dictionary for dumping
        """
        offset_dict = {}
        offset_dict["scale"]     = self.scale
        offset_dict["offset"]    = self.offset
        offset_dict["img_shift"] = self.offset * self.scale
        offset_dict["offset_global"] = self.offset_global
        return offset_dict
        
    def fromdict(self, offset_dict):
        """
        Updates the parameters from dict 
        """
        self.scale     = offset_dict["scale"]     
        self.offset    = offset_dict["offset"] 
        self.img_shift = offset_dict["img_shift"] 
        self.offset_global = offset_dict["offset_global"]
        if abs(self.img_shift - (self.offset * self.scale)) > 0.0000001:
            print("Inconsistency in the offset and scales")

