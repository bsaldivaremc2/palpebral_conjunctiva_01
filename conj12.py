"""
This library is an implementation of the feature extraction described in the paper:
Y.-M. Chen, S.-G. Miaou, and H. Bian, “Examining palpebral conjunctiva for anemia assessment with image processing methods,” 
Comput. Methods Programs Biomed., vol. 137, pp. 125–135, Dec. 2016.

Coded by franklin paul barrientos porras (frank2207a@gmail.com) 
and  Bryan Percy Saldivar Espinoza (bsaldivar.emc2@gmail.com)

Status of development: Under construction

It requires:
    opencv 3.2.0
    numpy 1.11.3
    matplotlib 2.0.0

==USAGE==
from  PIL import Image #To load an image
from conj12 import ConjFeat

img = 'img_test_.jpg'
imp = Image.open(img)
imn = np.asarray(imp)

#Create object specifying the image and channel order, default bgr, otherwise specify.
cf = ConjFeat(imn,input_channels='rgb')
#Extract the features. Set v=True for printing the generated values
values = cf.get_features(v=False)

#The features are returned as a dictionary
print(values.keys())
dict_keys(['HHR', 'Brightness', 'PVM', 'Entropy', 'PMV_12'])

"""
import numpy as np
import matplotlib.pyplot as plt
import cv2

class ConjFeat:
    def rgb_to_bgr(self,im):
        """
         This is a utility function to turn a RGB matrix into a BGR matrix.
        """
        om = np.zeros_like(im)
        om[:,:,2] = im[:,:,0]
        om[:,:,1] = im[:,:,1]
        om[:,:,0] = im[:,:,2]
        return om
    def __init__(self,image,hhr=50,input_channels='bgr'):
        """
        hhr is a value calculated manually in the paper. The default was 50.
        image is a bgr matrix of dimensions [n,m,3]. Atention! Not a RGB image but BGR.
        If you have a RGB matrix specify input_channels=='rgb' to re-arrange.
        """
        assert len(image.shape)==3

        self.hhr=hhr
        if input_channels=='rgb':
            self.image=self.rgb_to_bgr(image)
        else:
            self.image=image
        self.entropy=0
        self.HHR=0
        self.PVM   = np.zeros((3,1))
        self.area=0
        self.brightness = 0
        self.color = ('b','g','r')
        self.PVM_12= np.zeros((3,4))
    def calc_entropy(self,v=False):
        """
        The entropy is calculated using the green channel
        set v=True for information when it is calculated
        """
        equ = cv2.equalizeHist(self.image[:,:,1])
        hist,bins      = np.histogram(equ.flatten(),256,[0,256])
        norm_const     = hist.sum()
        dist_prob      = hist/norm_const
        for pixel in range(256):
            if dist_prob[pixel]!=0:
                self.entropy = self.entropy + np.abs(dist_prob[pixel]*np.log(dist_prob[pixel]))
        self.printv(v,"Entropy:",self.entropy)
    def calc_HHR(self,v=False):
        """
        Calculate the HHR feature
        set v=True for information when it is calculated
        """
        hsv    = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        for Tresh in range(self.hhr,257):
            self.HHR    = self.HHR + hsv[:,:,0].item(Tresh)/(self.image.shape[0]*self.image.shape[1])
        self.printv(v,"HHR:",self.HHR)
    def calc_PVM(self,v=False):
        """
        Calculate the PVM feature
        set v=True for information when it is calculated
        """
        Area   = np.shape(self.image)[0]*np.shape(self.image)[1];
        self.area=Area
        Ks     = round(0.4*Area)
        Ke     = round(0.6*Area)
        
        for i,col in enumerate(self.color):
            array_rgb = np.reshape(self.image[:,:,i], -1)
            array_sort= np.sort(array_rgb, axis=None)
            for value in range(Ks, Ke+1):
                self.PVM[i] = self.PVM[i] + array_sort[value]/(Ke-Ks+1)  
        self.printv(v,"PVM",self.PVM)
    def calc_bright(self,v=False):
        """
        Calculate the PVM feature
        set v=True for information when it is calculated
        """
        Brightness = np.sqrt(0.241*self.image[:,:,2] + 0.691*self.image[:,:,1] + 0.68*self.image[:,:,0])
        self.brightness = np.mean(Brightness)
        self.printv(v,"Brightness",self.brightness)
    def calc_PVMper(self,v=False):
        """
        Calculate PVM by percentile
        set v=True for information when it is calculated
        """
        Ks_l    = [round(0.022*self.area), round(0.5*self.area),round(0.88*self.area)]
        Ke_l    = [round(0.12*self.area), round(0.6*self.area), round(0.98*self.area)]
        for i,col in enumerate(self.color):
            array_rgb = np.reshape(self.image[:,:,i], -1)
            array_sort= np.sort(array_rgb, axis=None)
            for count in range(0,3):
                for value in range(Ks_l[count], Ke_l[count]+1):
                    self.PVM_12[i,count+1] = self.PVM_12[i,count+1] + array_sort[value]/(Ke_l[count]-Ks_l[count]+1)
            self.PVM_12[i,0] = np.mean(self.image[:,:,i])
        self.printv(v,"PVM-12 percentile",self.PVM_12)
    def calc_all_features(self,v=False):
        """
        Calculate all the features.
        set v=True to print output when it is calculated
        """
        self.calc_entropy(v)
        self.calc_HHR(v)
        self.calc_PVM(v)
        self.calc_bright(v)
        self.calc_PVMper(v)
    def get_features(self,v=False):
        """
        Return the features as a dictionary
        """
        self.calc_all_features(v)
        od={}
        od['PVM']=self.PVM
        od['PMV_12']=self.PVM_12,
        od['Brightness']=self.brightness
        od['HHR']=self.HHR
        od['Entropy']=self.entropy
        return od
    def printv(self,v,comment,value):
        if v==True:
            print(comment,value)