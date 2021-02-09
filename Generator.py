import pickle
import numpy as np
import tkinter as tk
from tkinter import *
from helpers import *
from tkinter import filedialog as fd
from tkinter import messagebox as mb

def loadConfig(fname):
    config = loadPickle(fname)
    for key in config.keys():
        globals()[str(key)] = config[key]
    globals()['configFile'] = fname
    
loadConfig('config.pkl')

class layerGenerator:
    def __init__(self, mode):
        
        self.size = None
        self.locs = None
        self.coeffs = None
        self.scalingFactor = None
        self.quantization_bits = None
        
        if 'retina' in mode.lower():
            self.mode = 'retina'
        elif 'cortex' in mode.lower():
            self.mode = 'cortex'
        else:
            print("Invalid packing mode selected!")
        
    def loadLocations(self, enc='latin1'):
        root = Tk()
        root.attributes("-topmost", True)
        root.withdraw()
        i = 0
        while self.locs is None:
            i+= 1
            try:
                fname = fd.askopenfilename(title = "Select locations data", filetypes = (("Pickle Object","*.pkl"),("all files","*.*")))
                with open(fname, "rb" ) as f:
                    self.locs = pickle.load(f , encoding=enc)
                if self.locs.shape[-1] != 7:
                    mb.showerror("Failed!", "Data has wrong dimensions.\nPlease make sure you have selected the correct file", parent=root)
                    self.locs = None
            except:
                self.locs = None
                mb.showerror("Failed!", "Something went wrong while trying to load location data.\nPlease make sure you have selected the correct file and that the file is not in use by other processes.", parent=root)
            if self.locs is None and i>=3:
                if mb.askyesno("Data not loaded correctly!", "Would you like to abort?", parent=root):
                    break
        if self.locs is None:
            raise Exception('Could not load location data!')
        root.destroy()
        w = 2*int(np.abs(self.locs[:,0]).max() + self.locs[:,6].max()/2.0)
        h = 2*int(np.abs(self.locs[:,1]).max() + self.locs[:,6].max()/2.0)
        self.size = np.array([h,w], dtype='int32')
        
    def loadCoefficients(self, enc='latin1'):
        root = Tk()
        root.attributes("-topmost", True)
        root.withdraw()
        i = 0
        while self.coeffs is None:
            i+= 1
            try:
                fname = fd.askopenfilename(title = "Select coefficients data", filetypes = (("Pickle Object","*.pkl"),("all files","*.*")))
                with open(fname, "rb" ) as f:
                    self.coeffs = np.squeeze(pickle.load(open(fname, "rb" ), encoding=enc))
                if len(self.coeffs.shape) != 1:
                    mb.showerror("Failed!", "Data has wrong dimensions.\nPlease make sure you have selected the correct file", parent=root)
                    self.coeffs = None
            except:
                self.coeffs = None
                mb.showerror("Failed!", "Something went wrong while trying to load coefficients data.\nPlease make sure you have selected the correct file and that the file is not in use by other processes.", parent=root)
            if self.coeffs is None and i>=3:
                if mb.askyesno("Data not loaded correctly!", "Would you like to abort?", parent=root):
                    break
        if self.coeffs is None:
            raise Exception('Could not load coefficients!')
        numFields = len(self.locs)+1

    def packPixels(self, toInts=True):
        ######################################
        ########### INITIALIZATION ###########
        ######################################
        h, w = self.size
        if self.mode == 'retina':
            self.locs[:,:2] = (self.locs[:,:2]+ np.array((w//2,h//2)))
        overlapCounter = np.zeros((h,w),dtype='uint8')
        #######################################
        ###### FINDING OVERLAPPING AREAS ######
        #######################################
        for field in range(len(self.locs)):
            size = self.locs[field][6]
            y1 = int(self.locs[:,1][field] - size/2+0.5)
            y2 = int(self.locs[:,1][field] + size/2+0.5)
            x1 = int(self.locs[:,0][field] - size/2+0.5)
            x2 = int(self.locs[:,0][field] + size/2+0.5)
            overlapCounter[y1:y2,x1:x2] += 1
        nLayers = np.max(overlapCounter)
        self.coeff_layers = []
        [self.coeff_layers.append(np.zeros((h,w))) for i in range(nLayers)]
        self.index_layers = []
        [self.index_layers.append(np.zeros((h,w))) for i in range(nLayers)]
        ######################################
        ########### PACKING PIXELS ###########
        ######################################
        for field in range(len(self.locs)):
            if self.mode == 'retina':
                idx = field+1
            elif self.mode == 'cortex':
                idx = int(self.locs[field,2])+1
            values = self.coeffs[field]
            size = int(self.locs[field][6])
            x1 = int(self.locs[:,0][field] - size/2+0.5)
            y1 = int(self.locs[:,1][field] - size/2+0.5)
            pixelCoords = [(int(x1+i), int(y1+j), values[j,i]) for j in range(size) for i in range(size)]
            for coord in pixelCoords:
                a , b , c = coord
                for i in range(nLayers):
                    if self.coeff_layers[i][b,a] == 0:
                        self.coeff_layers[i][b,a] = c
                        self.index_layers[i][b,a] = idx
                        break
        #######################################
        ###### CONVERTING FLOATS TO INTS ######
        #######################################
        if toInts:
            self.scalingFactor = ((2**self.quantization_bits)-1) / np.max(self.coeff_layers)
            for i in range(len(self.coeff_layers)):
                self.coeff_layers[i] = (self.coeff_layers[i]*self.scalingFactor).astype(numpy_types[types['COEFFICIENTS']])
                self.index_layers[i] = self.index_layers[i].astype(numpy_types[types['INDEX']])
        else:
            self.scalingFactor = 1
            
    def packKernels(self, toInts=True):
        ######################################
        ########### INITIALIZATION ###########
        ######################################
        h, w = self.size
        if self.mode == 'retina':
            self.locs[:,:2] = (self.locs[:,:2]+ np.array((w//2,h//2)))
        self.coeff_layers = []
        self.index_layers = []
        #######################################
        ########### PACKING KERNELS ###########
        #######################################
        for field in range(len(self.locs)):
            if self.mode == 'retina':
                idx = field+1
            elif self.mode == 'cortex':
                idx = int(self.locs[field,2])+1
            size = self.locs[field][6]
            y1 = int(self.locs[:,1][field] - size/2+0.5)
            y2 = int(self.locs[:,1][field] + size/2+0.5)
            x1 = int(self.locs[:,0][field] - size/2+0.5)
            x2 = int(self.locs[:,0][field] + size/2+0.5)
            found_empty_layer = False
            for x in range(len(self.coeff_layers)):
                coeff_layer = self.coeff_layers[x]
                index_layer = self.index_layers[x]
                if len(coeff_layer[np.where(coeff_layer[y1:y2,x1:x2]>0)])==0:
                    coeff_layer[y1:y2,x1:x2] = self.coeffs[field]
                    index_layer[y1:y2,x1:x2] = idx
                    found_empty_layer = True
                    break
            if not found_empty_layer:
                self.coeff_layers.append(np.zeros((h,w)))
                self.index_layers.append(np.zeros((h,w)))

                self.coeff_layers[-1][y1:y2,x1:x2] = self.coeffs[field]
                self.index_layers[-1][y1:y2,x1:x2] = field+1
        #######################################
        ###### CONVERTING FLOATS TO INTS ######
        #######################################
        if toInts:
            self.scalingFactor = ((2**self.quantization_bits)-1) / np.max(self.coeff_layers)
            for i in range(len(self.coeff_layers)):
                self.coeff_layers[i] = (self.coeff_layers[i]*self.scalingFactor).astype(numpy_types[types['COEFFICIENTS']])
                self.index_layers[i] = self.index_layers[i].astype(numpy_types[types['INDEX']])
        else:
            self.scalingFactor = 1

        
    def save(self, outputFile):
        with open(outputFile, "wb") as f:
            pickle.dump((self.coeff_layers,self.index_layers,self.scalingFactor), f)
        config = {'types' : types,
                  'numpy_types' : numpy_types,
                  'numFields' : numFields}
        savePickle(configFile, config)