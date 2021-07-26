#!/usr/bin/env python3
"""
Probability distributions
"""
import copy as cp
import numpy as np
import math as mt

class probability_dist():
    """
    Class to handle probability distributions
    """
    def __init__(self):
        """
        Initialize things
        """

    def gaussian(self, x,height,position,FWHM,beta):
        """
        Basic gaussian function expressed with respect to FWHM
        """
        std = FWHM*0.5*np.sqrt(2*np.log(2))
        return height*np.exp(-0.5*np.power(np.abs(x-position)/std,beta))

    def normalize(self, dist):
        """
        Normalizes a distribution that the sum of the probabilities is one
        """
        ddist = dist/sum(dist.flatten())
        return ddist
    
    def sum(self, dist_list, name, weight):
        """
        Returns a single normalized probability distribution, where every
        entry of the list is a dictionary with a "weight" and a "dist" that is
        transformed into a weighted sum, correcly normalized
        """
        if (len(dist_list)==0):
            print("Empty list of distributions")
            return
        e = cp.deepcopy(dist_list[0])
        ddist = e["weight"]*e["dist"]
        for e in dist_list[1:]:
            ddist += e["weight"]*e["dist"]
        d = {}
        d["weight"] = weight
        d["dist"] = self.normalize(ddist)
        d["name"] = name
        return d
    
    def prod(self, dist_list, name, weight):
        """
        Returns a single normalized probability distribution, where every
        entry of the list is a dictionary with a "weight" and a "dist" that is
        transformed into a weighted product, correcly normalized
        """
        if (len(dist_list)==0):
            print("Empty list of distributions")
            return
        e = cp.deepcopy(dist_list[0])
        ddist = e["weight"]*e["dist"]
        for e in dist_list[1:]:
            ddist *= e["weight"]*e["dist"]
        d = {}
        d["weight"] = weight
        d["dist"] = self.normalize(ddist)
        d["name"] = name
        return d
    
    def make(self, name, weight, dist):
        """
        Returns a dictionary of normalized pd
        """
        d = {}
        d["name"] = name
        d["weight"] = weight
        d["dist"] = self.normalize(dist)
        return d
    
    def collapse(self, dist, sampled):
        """
        Reduces the dimensionality of a distribution dist to the one of sampled, retaining
        the probabilities
        """
        d = np.zeros(len(sampled))
        r = float(len(dist))/float(len(sampled))
        for i in range(len(sampled)):
            d[i] = sum(dist[int(i*r) : int((i+1)*r)])
        return d
    
    def select_max(self, dist, sampled):
        """
        Returns the index of maximal probability, given it has not yet been
        sampled
        Assume both arrays have same dim
        """
        if (len(dist)!=len(sampled)):
            print("Dimensions messed up")
            return
        prob = dist*sampled
        return np.argmax(prob)
    
    def select_random(self, dist, sampled):
        """
        Returns a random index given the distribution in dist, but taking into
        account the sampled points
        sampled
        Assume both arrays have same dim
        """
        if (len(dist)!=len(sampled)):
            print("Dimensions messed up")
            return
        prob = dist*sampled
        prob = prob/sum(prob)
        return np.random.choice(len(sampled),size=1,replace=False,p=prob)

    def gaussian(self, x, height, position, FWHM, beta) :
        """
        Evaluates a gaussian
        """
        std = FWHM * 0.5 * np.sqrt(2 * np.log(2))
        return height * np.exp(-0.5 * np.power(np.abs(x - position)/std, beta))
    
    def convert_gaussianbias(self, bias_parameters):
        """
        Returns a set of gaussian parameters in a format the the Sara GP can
        handle
        """
        bias_parameters_new = []
        for b in bias_parameters:
        #for b in [bias_parameters[-1]]: #Only the last parameter
            std = b[2] * 0.5 * np.sqrt(2.*np.log(2.))
            bb = (b[0] * b[4], b[1], std, b[3])
            bias_parameters_new.append(bb)
        return bias_parameters_new

    def get_img_bias(self, peaks, img_center, spec_center, x_plot, lsa_frac = 0.75, fwhm_peaks = 0.02, weight = 0.02):
        """
        Create the probability distributions based on the RGB peaks
        Here we use a fixed FWHM, but in the future we can use others
        """
        #RGB data here
        dist_peaks = []
        bias_parameters = []
        #fwhm_peaks = 0.02 #in mm
        #weight = 0.02
        #Symmetrize the peaks here
        peaks_sym = cp.deepcopy(peaks).tolist()
        for i in peaks:
            peaks_sym.append(-(i - img_center) + img_center)
        for i in peaks_sym:
            dist = self.gaussian(x_plot.reshape((-1,1)), 1., i - img_center + spec_center, fwhm_peaks,2)
            dist_peaks.append(self.make(str(i), weight, dist))
            bias_parameters.append(np.array([1., i - img_center + spec_center, fwhm_peaks, 2., weight]))

        """
        Create the probability distribution based on the zone width
        The width of the sampling gaussian is taken from the min/max peaks
        of the image
        """
        LSA_width = 0.5
        LSA_weight = 1.
        if (len(peaks)> 2):
            LSA_width = max(lsa_frac * (np.max(peaks_sym) - np.min(peaks_sym)), LSA_width)
        dist_lsa = self.gaussian(x_plot.reshape((-1,1)), 1., spec_center, LSA_width, 8)
        bias_parameters.append(np.array([1., spec_center, LSA_width, 8., LSA_weight]))
        dist_peaks_lsa = cp.deepcopy(dist_peaks)
        dist_peaks_lsa.append(self.make("Gauss", 1., dist_lsa))
        return dist_peaks, dist_lsa, dist_peaks_lsa, bias_parameters, LSA_width


