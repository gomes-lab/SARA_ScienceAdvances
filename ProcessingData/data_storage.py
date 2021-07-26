"""
Data handling for images and reflectance
"""

from gzopen import gzopen
import logging
import logging.config
import yaml
import copy as cp
import re
import ntpath
import h5py
import datetime
import numpy as np
import time
import io
import glob
import pandas as pd
import matplotlib.pyplot as plt


class wafer():
    """
    Class representing a wafer
    """

    def __init__(self):
        self.sample_id = None
        self.sample_description = None
        self.logger = logging.getLogger("Wafer")
        self.logger.setLevel(logging.INFO)
        self.score = lambda i: ("+" if i > 0 else "") + str(i)

    def fft_smoothing(self, d, param):
        """
        Helper functions
        d is a 1D vector, param is the cutoff frequency
        """
        rft = np.fft.rfft(d)
        rft[int(param):] = 0.
        d = np.fft.irfft(rft, len(d))
        return d

class LSA(wafer):
    """
    Subclass to handle reading, writing, and logging LSA data
    """

    def __init__(self):
        super(LSA, self).__init__()
        self.logger = logging.getLogger("WaferLSA")
        self.data_type =  "LSA"
        self.img_prefix = "b"
        self.spec_prefix = "s"
    
    def coord_format(self, d, l):
        q = str(int(abs(d))).zfill(l)
        if d >= 0.:
            q = "+" + q
        else:
            q = "-" + q
        return q

    def read(self, fn):
        """
        Reads the positon, dwell, and peak temperature of a processed wafer
        """
        self.logger.info("Reading %s", fn)
        data_list = []
        #data = np.genfromtxt(fn, dtype=float, delimiter=',', skip_header=1)
        data = pd.read_csv(fn, delimiter=',', skiprows=1, header=None).values
        for i in range(data.shape[0]):
            data_dict = {"pos"  : data[i, 0:2].tolist(), 
                         "x"    : data[i, 0], 
                         "y"    : data[i, 1], 
                         "dwell": data[i, 2],
                         "Tpeak": data[i, 3]}
            try:
                data_dict["Power"] = data[i, 4]
            except:
                self.logger.debug("No power value available")
            data_list.append(data_dict)
            self.image_name(data_dict)
        return data_list

    def image_name(self, stripe, suffix = None):
        """
        Returns the (presumable) file name given the information from the LSA process log
        """
        name  = self.img_prefix
        #name += "_"
        name += self.coord_format(stripe["x"], 2) 
        name += "_"
        name += self.coord_format(stripe["y"], 2) 
        name += "_"
        name += self.coord_format(stripe["dwell"], 5) 
        name += "_"
        name += self.coord_format(stripe["Tpeak"], 4) 
        if suffix is not None:
            name += "_"
            name += suffix
        name += ".bmp"
        return name

    def spec_name(self, stripe, suffix = None):
        """
        Returns the (presumable) file name given the information from the LSA process log
        """
        name  = self.spec_prefix
        #name += "_"
        name += self.coord_format(stripe["x"], 2) 
        name += "_"
        name += self.coord_format(stripe["y"], 2) 
        name += "_"
        name += self.coord_format(stripe["dwell"], 5)
        name += "_"
        name += self.coord_format(stripe["Tpeak"], 4) 
        if suffix is not None:
            name += "_"
            name += suffix
        name += ".csv"
        return name

    def spec_name_old(self, stripe):
        """
        Returns the (presumable) file name given the information from the LSA process log
        """
        name  = self.spec_prefix
        #name += "_"
        name += self.coord_format(stripe["x"], 2) 
        name += "_"
        name += self.coord_format(stripe["y"], 2) 
        name += "_"
        name += str(int(stripe["dwell"])).zfill(5)
        name += "_"
        name += str(int(stripe["Tpeak"])).zfill(4)
        name += ".csv"
        return name

class spec(wafer):
    """
    Subclass to handle reading, writing, and logging spectroscopy data
    """
    def __init__(self):
        super(spec, self).__init__()
        self.logger = logging.getLogger("WaferSpec")
        self.data_type =  "Spec"
        self.dead_fix = True
        self.dead_pixel = 1388
        self.smooth = 15

    def user_coord(self, data_dict):
        """
        Given a stripe of specs, split into individual specs
        with user coordinates 
        """
        spect_dict = [] 
        xmin = data_dict["Meta"]["Position"]["Range"][0][0] + data_dict["Meta"]["Position"]["Center"][0]
        xmax = data_dict["Meta"]["Position"]["Range"][1][0] + data_dict["Meta"]["Position"]["Center"][0]
        ymin = data_dict["Meta"]["Position"]["Range"][0][1] + data_dict["Meta"]["Position"]["Center"][1]
        ymax = data_dict["Meta"]["Position"]["Range"][1][1] + data_dict["Meta"]["Position"]["Center"][1]
        nlines = data_dict["Meta"]["Position"]["Scan lines"]
        x_list = np.linspace(xmin, xmax, nlines).tolist()
        y_list = np.linspace(ymin, xmax, nlines).tolist()
        for x, y, i in zip(x_list, y_list, range(len(x_list))):
            d = {"pos"          : [x, y],
                 "x"            : x,
                 "y"            : y,
                 "Tpeak"        : data_dict["Meta"]["Tpeak"],
                 "dwell"        : data_dict["Meta"]["dwell"],
                 "Spec"         : data_dict["Spec"][:, i],
                 "Wavelengths"  : data_dict["Wavelengths"] 
                }
            spect_dict.append(d)
        return spect_dict

    def init_meta(self):
        """
        Initialize metadata here
        """
        #Structure of metadata defined here
        meta = {
             "Timestamp": None,
             "Position": None,
             "Ocean Optics Spectrometer": None, 
             "Collection": None,
             "Tpeak": None,
             "dwell": None
        }
        return meta

    def read_old_meta(self, fn, meta):
        """
        Reading new metadata from file, the old way
        """
        position = {}
        with gzopen(fn) as f:
            first_line = f.readline().strip().split(' ')
            first_line = [elem.replace("(","").replace(")","") for elem in first_line]
            position['Center'] = [float(i) for i in first_line[3].split(',')]
            rng = []
            rng.append([float(i) for i in first_line[7].split(',')])
            rng.append([float(i) for i in first_line[11].split(',')])
            position['Range'] = rng
            position['Scan lines'] = int(first_line[14])
        if position in locals():
            meta["Position"] = position
        f.close()
        return meta

    def read_new_meta(self, fn, meta):
        """
        Reading new metadata from file
        """
        position = None
        spectrometer = None
        collection = None
        with gzopen(fn) as f:
            cnt = 0
            for line in f:
                if line.startswith('/'):
                    if "Position" in line:
                        position = {}
                    if "Ocean Optics Spectrometer" in line:
                        spectrometer = {}
                    if "Collection" in line:
                        collection = {}
                    if "Timestamp" in line:
                        timestamp = int(line.strip().split()[2])
                        meta["Timestamp"] = timestamp
                    if "Center" in line:
                        position['Center'] = [float(i) for i in line.strip().split()[-1].replace("(", "").replace(")", "").split(',')]
                    if "Range" in line:
                        rng = []
                        l = line.strip().split()[-2:]
                        l = [elem.replace("(","").replace(")","") for elem in l]
                        rng.append([float(i) for i in l[0].split(',')])
                        rng.append([float(i) for i in l[1].split(',')])
                        position['Range'] = rng
                    if "Scan lines" in line:
                        position['Scan lines']  = int(line.strip().split()[-1])
                    if "Focus" in line:
                        position['Focus']       = float(line.strip().split()[-1])
                    if "Model" in line:
                        spectrometer["Model"]   = line.strip().split()[-1]
                    if "S/N" in line:
                        spectrometer["S/N"]     = line.strip().split()[-1]
                    if "API" in line:
                        spectrometer["API"]     = line.strip().split()[-1]
                    if "Shutter" in line:
                        spectrometer["Shutter"] = float(line.strip().split()[-2])
                    if "Averages" in line:
                        collection["Averages"] = int(line.strip().split()[-1])
                    if "Dark pixel correction" in line:
                        collection["Dark pixel correction"] = str(line.strip().split()[-1])
                    if "Non-linear correction" in line:
                        collection["Non-linear correction"] = str(line.strip().split()[-1])
                cnt += 1
        if position != None:
            meta["Position"] = position
        if spectrometer != None:
            meta["Ocean Optics Spectrometer"] = spectrometer
        if collection != None:
            meta["Collection"] = collection
        f.close()
        return meta

    def write_csv_meta_header(self, meta):
        """
        Returns a string formatted according to the metadata format the genplot (or Mike in general) produces
        """
        meta_string  = "/* Block scan\n"
        dt = time.ctime(int(meta["Timestamp"]))
        key = "Timestamp"   ; meta_string += "/* %s: %s %s\n"               %(key, meta[key], dt) 
        key = "Position"    ; meta_string += "/* %s:\n"                     %(key)
        key_1 = "Center"    ; meta_string += "/*  %s: (%12.8f, %12.8f)\n"   %(key_1, meta[key][key_1][0], meta[key][key_1][1])
        key_1 = "Range"     ; meta_string += "/*  %s: (%12.8f, %12.8f)  (%12.8f, %12.8f)\n" %(key_1, meta[key][key_1][0][0], meta[key][key_1][0][1], meta[key][key_1][1][0], meta[key][key_1][1][1])
        key_1 = 'Scan lines'; meta_string += "/*  %s: %8d \n"               %(key_1, meta[key][key_1])
        key_1 = "Focus"     ; meta_string += "/*  %s: %12.8f\n"             %(key_1, meta[key][key_1])
        key = "Ocean Optics Spectrometer" ; meta_string += "/* %s:\n"       %(key)
        key_1 = 'Model'     ; meta_string += "/*  %s: %s\n"                 %(key_1, meta[key][key_1])
        key_1 = 'S/N'       ; meta_string += "/*  %s: %s\n"                 %(key_1, meta[key][key_1]) 
        key_1 = 'API'       ; meta_string += "/*  %s: %s\n"                 %(key_1, meta[key][key_1])  
        key_1 = 'Shutter'   ; meta_string += "/*  %s: %15.8f\n"             %(key_1, meta[key][key_1])   
        key = "Collection"  ; meta_string += "/* %s:\n"       %(key)
        key_1 = 'Averages'              ; meta_string += "/*  %s: %s\n"     %(key_1, meta[key][key_1]) 
        key_1 = 'Dark pixel correction' ; meta_string += "/*  %s: %s\n"     %(key_1, meta[key][key_1]) 
        key_1 = 'Non-linear correction' ; meta_string += "/*  %s: %s\n"     %(key_1, meta[key][key_1]) 
        return meta_string

    def write_csv_meta_header_reference(self, meta):
        """
        Returns a string formatted according to the metadata format the genplot (or Mike in general) produces
        """
        meta_string  = "/* OceanOptics spectrum\n"
        dt = time.ctime(int(meta["Timestamp"]))
        key = "Timestamp"   ; meta_string += "/* %s: %s %s\n"               %(key, meta[key], dt) 
        key = "Ocean Optics Spectrometer" ; meta_string += "/* %s:\n"       %(key)
        key_1 = 'Model'     ; meta_string += "/*  %s: %s\n"                 %(key_1, meta[key][key_1])
        key_1 = 'S/N'       ; meta_string += "/*  %s: %s\n"                 %(key_1, meta[key][key_1]) 
        key_1 = 'API'       ; meta_string += "/*  %s: %s\n"                 %(key_1, meta[key][key_1])  
        key_1 = 'Shutter'   ; meta_string += "/*  %s: %15.8f\n"             %(key_1, meta[key][key_1])   
        key = "Collection"  ; meta_string += "/* %s:\n"       %(key)
        key_1 = 'Averages'              ; meta_string += "/*  %s: %s\n"     %(key_1, meta[key][key_1]) 
        key_1 = 'Dark pixel correction' ; meta_string += "/*  %s: %s\n"     %(key_1, meta[key][key_1]) 
        key_1 = 'Non-linear correction' ; meta_string += "/*  %s: %s\n"     %(key_1, meta[key][key_1]) 
        meta_string += "/* wavelength [nm] , raw\n"
        return meta_string

    def write_csv_data(self, wavelengths, spectra_data_list):
        """
        Returns a string for writing the data with wavelength first, followed by
        columns of measured stripe data
        """
        data_string  = "/* Data\n"
        wl = np.array(wavelengths).flatten()
        sp = np.array(spectra_data_list)
        if wl.shape[0] == sp.shape[0]:
            data_list = [wl.tolist()] + sp.T.tolist()
            if sp.ndim == 1:
                data_list = [wl.tolist()] + [sp.tolist()]
        else:
            data_list = [wl.tolist()] + sp.tolist()
        data_array = np.array(data_list).T
        s = io.StringIO()
        np.savetxt(s, data_array, delimiter=',', fmt='%f')
        data_string += s.getvalue()
        return data_string

    def write_csv_spec(self, fn_data, meta, wavelengths, spectra_data_list):
        """
        Writes spectra csv file
        """
        with open(fn_data, 'w') as f:
            f.write(self.write_csv_meta_header(meta))
            f.write(self.write_csv_data(wavelengths, spectra_data_list))

    def write_csv_spec_reference(self, fn_data, meta, wavelengths, spectra_data_list):
        """
        Writes spectra csv file for mirror and blank
        """
        with open(fn_data, 'w') as f:
            f.write(self.write_csv_meta_header_reference(meta))
            f.write(self.write_csv_data(wavelengths, spectra_data_list))

    def read_spec(self, fn_data):
        """
        Read lasgo-type spectroscopy scan files
        """
        meta = self.init_meta()
        self.logger.info("Reading file %s", fn_data)
        self.newversion = False
        with gzopen(fn_data) as f:
            cnt = 0
            for line in f:
                if line.startswith('/'):
                    if "correction" in line:
                        self.newversion = True
                cnt += 1
        f.close()

        if self.newversion:
            #New format of files for reading
            self.logger.debug("Reading new file format")
            #Parse header and metadata
            meta = self.read_new_meta(fn_data, meta)
            #data = np.genfromtxt(fn_data, dtype=float, delimiter=',', skip_header=17)
            data = pd.read_csv(fn_data, delimiter=',', skiprows=17, header=None).values
        else:
            #Old format of files for reading
            self.logger.debug("Reading old file format")
            #Parse header and metadata
            meta = self.read_old_meta(fn_data, meta)
            #data = np.genfromtxt(fn_data, dtype=float, delimiter=',', skip_header=1)
            data = pd.read_csv(fn_data, delimiter=',', skiprows=1, header=None).values
        self.logger.info("Read file %s", fn_data)

        #Parse information from the filename
        try:
            fn_base = ntpath.basename(fn_data)
            fn_meta = fn_base.split("_")
            #The last part is the temperature in C
            meta["Tpeak"] = float(fn_meta[-1].split(".")[0])
            #The second last part is the temperature in dwell time in microsec
            meta["dwell"] = float(fn_meta[-2])
            #We can check the coordinates with the ones already read from the comment block
            dy = float(fn_meta[-3])
            dx = float(re.sub('[^0-9,+,-]','',fn_meta[-4].split("/")[-1]))
            dd = np.linalg.norm(np.array([dx,dy])-np.array(meta['Position']["Center"]))
            if dd > 0.0001:
                self.logger.warning("The coordinate from the metadata and the filename disagree %f", dd)
        except:
            self.logger.warning("Could not retrieve metadata from filename")

        #Getting rid of a dead pixel
        if self.dead_fix:
            data[self.dead_pixel, 1:]  = (data[self.dead_pixel-1, 1:] + data[self.dead_pixel+1, 1:]) * 0.5
            self.logger.info("Eliminated dead pixel at %d", self.dead_pixel)

        #Wavelengths
        wl = data[:,0]
        data = np.array(data)[:,1:]
        data_dict = {"Wavelengths": wl, "Spec": data, "Meta": meta} 
        return data_dict

    def read_mirror_blank(self, fn_data):
        """
        Read lasgo-type spectroscopy scan files, at the mirror or blank
        """
        meta = self.init_meta()
        spectrometer = {}
        collection = {}

        self.logger.info("Reading file %s", fn_data)
        self.newversion = False
        with gzopen(fn_data) as f:
            cnt = 0
            for line in f:
                if line.startswith('/'):
                    if "correction" in line:
                        self.newversion = True
                cnt += 1
        f.close()

        if self.newversion:
            #New format of files for reading
            self.logger.debug("Reading new file format")
            #Parse header and metadata
            meta = self.read_new_meta(fn_data, meta)
            #data = np.genfromtxt(fn_data, dtype=float, delimiter=',', skip_header=13) #Mike calls it ref
            data = pd.read_csv(fn_data, delimiter=',', skiprows=13, header=None).values
        else:
            #Old format of files for reading
            self.logger.debug("Reading old file format")
            #Parse header and metadata
            meta = self.read_old_meta(fn_data, meta)
            #data = np.genfromtxt(fn_data, dtype=float, delimiter=',', skip_header=1) #Mike calls it ref
            data = pd.read_csv(fn_data, delimiter=',', skiprows=1, header=None).values
        self.logger.info("Read file %s", fn_data)

        #Getting rid of a dead pixel
        if self.dead_fix:
            data[self.dead_pixel,1]  = (data[self.dead_pixel-1, 1]+data[self.dead_pixel+1, 1]) * 0.5
            self.logger.info("Eliminated dead pixel at %d", self.dead_pixel)

        #Wavelengths
        wl = data[:,0]
        data = np.array(data)[:,1]
        data_dict = {"Wavelengths": wl, "Spec": data, "Meta": meta} 
        return data_dict

    def write_mirror_blank(self, fn_data):
        return

    def normalize(self, data, mirror, blank):
        scaling = 1.
        start = time.time()
        #Smooth data
        mirror_s = self.fft_smoothing(mirror[:], self.smooth)
        blank_s = self.fft_smoothing(blank[:], self.smooth)
        data_s = cp.deepcopy(data)
        if len(data.shape) == 1:
            data_s[:] = self.fft_smoothing(data[:], self.smooth)
        else:
            for i in range(data.shape[1]):
                data_s[:,i] = self.fft_smoothing(data[:,i], self.smooth)
        #Normalization
        #norm = np.maximum(np.ones(mirror_s.shape), mirror_s[:] - blank_s[:])
        norm = mirror_s[:] - blank_s[:]
        normal_data = cp.deepcopy(data_s)
        #normal_data = ((np.maximum(data_s, 1.) - blank_s[:].reshape(-1, 1))/norm.reshape(-1, 1)) * scaling
        normal_data = ((data_s - blank_s[:].reshape(-1, 1))/norm.reshape(-1, 1)) * scaling
        end = time.time()
        return normal_data

    def convert_spectrometer_dict(self, spectrometer_dict):
        """
        Returns meta information in Mike's format from the spectrometer
        """
        data_dict = spectrometer_dict
        info_o = {}
        info_o["Model"] = data_dict['model']
        info_o["S/N"] = data_dict['serial']
        info_o["API"] = "None"
        info_o["Shutter"] = data_dict['ms_integrate']
        return info_o, data_dict
    
    def convert_spec_dict(self, spectrum_dict):
        """
        Returns meta information in Mike's format from the spectrum
        """
        data_dict = spectrum_dict
        info_c = {}
        info_c["Averages"] = data_dict['num_average']
        if data_dict['use_dark_pixel'] == 1:
            info_c["Dark pixel correction"] = "On"
        else:
            info_c["Dark pixel correction"] = "Off"
        if data_dict['use_nl_correct'] == 1:
            info_c["Non-linear correction"] = "On"
        else:
            info_c["Non-linear correction"] = "Off"
        return info_c, data_dict
    
    def convert_scan_info(self, zone, pos_list, focus_list):
        """
        Returns meta information of a stripe scan
        """
        info_p = {}
        info_p["Center"] = [zone["x"], zone["y"]]
        info_p["Range"]  = [[pos_list[0][0] - info_p["Center"][0], pos_list[0][1] - info_p["Center"][1]], 
                [pos_list[-1][0] - info_p["Center"][0], pos_list[-1][1] - info_p["Center"][1]]]
        info_p["Scan lines"]  = len(pos_list)
        info_p["Focus"]  = np.average(focus_list)
        return info_p
    
    def convert_meta(self, pos_list, focus_list, spectrometer_dict, spectrum_dict, zone = None):
        """
        Returns complete meta information to save in Mike's stripe format
        """
        meta = {}
        info_o, raw_spectrometer_info = self.convert_spectrometer_dict(spectrometer_dict)
        info_c, raw_spec_info = self.convert_spec_dict(spectrum_dict)
        meta["Timestamp"] = raw_spec_info['timestamp']
        if zone is not None:
            info_p = self.convert_scan_info(zone, pos_list, focus_list)
            meta["Position"] = info_p
        meta["Ocean Optics Spectrometer"] = info_o
        meta["Collection"] = info_c
        return meta

    def convert_meta_reference(self, spectrometer_dict, spectrum_dict):
        """
        Returns complete meta information to save in Mike's stripe for mirror
        and blank
        """
        meta = {}
        info_o, raw_spectrometer_info = self.convert_spectrometer_dict(spectrometer_dict)
        info_c, raw_spec_info = self.convert_spec_dict(spectrum_dict)
        meta["Timestamp"] = raw_spec_info['timestamp']
        meta["Ocean Optics Spectrometer"] = info_o
        meta["Collection"] = info_c
        return meta
        
