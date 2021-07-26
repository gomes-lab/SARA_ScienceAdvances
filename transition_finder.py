import glob
import os
import cv2
from PIL import Image
import numpy as np
from signalsmooth import *
from scipy import ndimage

def HSI_Tr_finder2(filepath,mpath,blur=5,Gpromfilt=200,c_thresh=0.025,v_cutoff=0,h_thresh=4.0,plotting=True,norm=True,ImgIO=False,pix_siz = 1.0319258564814815):
    """Determines the transitions of the HSI images and outputs a metadata
    
    filepath:  (path) location to file OR Image data, see specifier
    mpath:     (path) location to mirror file
    blur:      (int) the amount of blur applied to smooth the image
    Gpromfilt: (int) the threshold value that determines which peaks are accepted in the gradient of the summed image
    c_thresh:  (float) a value between 0 and 1 that determines whether or not transitions are recorded
    v_cutoff   (int) an index value corresponding to the number of pixels shaved off the top and bottom of the image 
    h_thresh:  (float) adds a height threshold valule based on the "unannealed" portion of the stripe
    plotting:  (bool) displays plots that contain the image, where the transitions occur, the gradient, and which peaks are involved
    norm:      (bool) applys histogram filling of an image to ACB and divides the image by the mirror     
    ImageIO:   (bool) This allows one to use the "filepath" and "mpath" as array inputs for your data
    
    (future edits)
    Could also use symmetry constraints
    
    Returns: (dict) Output dictionary containing transitions and transition finding paramenters.
    
    
    """
    
    TransitionStats = {}
    outputfile = {}
    arr_output = []

    #this is for labeles on plots and key construction
    if ImgIO == False:
        file = glob.glob(filepath)[0]
        filename = os.path.basename(file)[:-4]
        cond = 'filler'
        filename = 'filler'
    else:
        cond = 'filler'
        filename = 'filler'
        file = 'filler'
        
        
    #This is where the imaeg processing starts
    if 'align' not in filename:
        if ImgIO==True:
            img1 = filepath[v_cutoff:(filepath.shape[0]-v_cutoff),:]
            img = cv2.equalizeHist(img1.astype('uint8'))
            img = Image.fromarray(img.astype('uint8'))
            if norm==True:
                bkg = mpath[v_cutoff:(filepath.shape[0]-v_cutoff),:]
                img = cv2.equalizeHist(((img1/bkgd)/np.max((img1/bkgd))*255).astype('uint8'))
                img = Image.fromarray(img,mode='L')
            else:
                img = cv2.equalizeHist(img1.astype('uint8'))
                img = Image.fromarray(img,mode='L')
                
                
        else:
            img1 = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            img1 = img1[v_cutoff:(img1.shape[0]-v_cutoff),:]
            
            if norm==True:
                bkgd = np.array(Image.open(mpath,'r'))
                bkgd = bkgd[v_cutoff:(bkgd.shape[0]-v_cutoff),:]

                img = cv2.equalizeHist(((img1/bkgd)/np.max((img1/bkgd))*255).astype('uint8'))
                img = Image.fromarray(img,mode='L')
            else:
                img = cv2.equalizeHist(img1.astype('uint8'))
                img = Image.fromarray(img,mode='L')
            
        xpix = np.array(img).shape[1]
        ypix = np.array(img).shape[0]
        
        fp = blur/100

        #calling on a partiular channel
        HSI_1D = smooth(np.sum(ndimage.gaussian_filter(img,(6,6)),axis=0),window_len=np.round(fp*img.size[0]).astype(int))
        
        
        HSI_mean = np.mean(HSI_1D)
        HSI_std = np.std(HSI_1D)
        HSI_contrast = HSI_std/HSI_mean
        
        Grad = np.gradient(HSI_1D)
        Grad = np.sqrt(Grad*Grad)
        
        h_thresh = h_thresh*((np.mean(Grad[0:100])+np.mean(Grad[-100:]))*0.5)
        #removing DC component
        HSI_1D = (HSI_1D-HSI_mean)/HSI_std

        #peak finding in the gradient I used an all encompasing sest so I could get the fit values to output
        #same with the height
        HSI_trans = signal.find_peaks(Grad,prominence=Gpromfilt,width=[0.001*xpix,0.25*xpix],height=h_thresh)
        
        #Gradient correlation
        correlation = ndimage.correlate(Grad,Grad[::-1],mode='wrap')
        HSI_center = (len(correlation))*0.5 + (np.argmax(correlation)-(len(correlation))*0.5)*0.5
    
        TransitionStats['center'] = HSI_center
        TransitionStats['mean'] = HSI_mean
        TransitionStats['std'] = HSI_std
        TransitionStats['contrast'] = HSI_contrast
        TransitionStats['S/N'] = HSI_mean/HSI_std
        TransitionStats['filepath'] = filepath

        if HSI_contrast>c_thresh:
            #print('transitions are here')
            #print(HSI_contrast,c_thresh)
            TransitionStats['TR_idx'] = HSI_trans[0]
            TransitionStats['transitions'] = {}
            TransitionStats['transitions']['widths'] = HSI_trans[1]['widths']
            TransitionStats['transitions']['heights'] = HSI_trans[1]['peak_heights']
            TransitionStats['transitions']['S/N'] = HSI_trans[1]['peak_heights']/HSI_std
            TransitionStats['transitions']['prominences'] = HSI_trans[1]['prominences']
            TransitionStats['transitions']['distance_from_center'] = (HSI_trans[0]-HSI_center)*pix_siz 
            TransitionStats['StripeCenterDistFromImageCenter']=(HSI_center-int(0.5*img.size[0]))*pix_siz
            TransitionStats['horizontal FOV']=xpix*pix_siz
            TransitionStats['vertical FOV']=ypix*pix_siz
            TransitionStats['LSA_condition']=cond
        else:
            TransitionStats['transitions']='none'
            
        outputfile=TransitionStats
        
        
        if plotting == True:
            plt.figure()
            plt.title('Anneal conditions '+cond)
            plt.imshow(img1,cmap='Greys',aspect='auto')
            plt.colorbar()
            if HSI_contrast>c_thresh:
                for tr in HSI_trans[0]:
                    plt.plot(tr*np.ones(int(img.size[1]/2)),np.arange(int(img.size[1]/2)),c='gold')
            plt.axvline(x=HSI_center,ymin=0.5,ymax=1,c='darkorchid',linestyle=':')
            plt.show()
            
            plt.figure()
            plt.title('Gradient plot')
            plt.plot(Grad,'r')
            if HSI_contrast>c_thresh:
                for tr in HSI_trans[0]:
                    plt.axvline(x=tr,ymax=1,ymin=0.5,c='gold')
            plt.show()
        return outputfile

