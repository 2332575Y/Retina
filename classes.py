import numpy as np
from helpers import *
from functions import *

def loadConfig(fname):
    config = loadPickle(fname)
    for key in config.keys():
        globals()[str(key)] = config[key]
    globals()['configFile'] = fname
    
loadConfig('config.pkl')

#===============================================================================================================================
#===============================================================================================================================
#                                         ██████╗ ███████╗████████╗██╗███╗   ██╗ █████╗ 
#                                         ██╔══██╗██╔════╝╚══██╔══╝██║████╗  ██║██╔══██╗
#                                         ██████╔╝█████╗     ██║   ██║██╔██╗ ██║███████║
#                                         ██╔══██╗██╔══╝     ██║   ██║██║╚██╗██║██╔══██║
#                                         ██║  ██║███████╗   ██║   ██║██║ ╚████║██║  ██║
#                                         ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝
#===============================================================================================================================
#===============================================================================================================================
class Retina:
    def __init__(self, fname):
        self.backProjectedVector = None
        self.normalizationVector = None
        self.input_resolution = None
        self.scalingFactor = None
        self.sampledVector = None
        self.coeff_layers = None
        self.index_layers = None
        self.layersFile = fname
        self.crop_coords = None
        self.backProject = None
        self.getResult = None
        self.fixation = None
        self.sample = None
        self.size = None

        try:
            self.loadLayers()
        except:
            raise Exception('Could not find previously saved layers, Please make sure to initlaize them!')
            

    def loadLayers(self):
        self.coeff_layers, self.index_layers, self.scalingFactor = loadPickle(self.layersFile)
        self.sampledVector = np.zeros(numFields, dtype=numpy_types[types['RESULTS']])
        
    def setInputResolution(self, w, h):
        self.input_resolution = np.array([h,w], dtype='int32')
        
    ########################################
    ############### FIXATION ###############
    ########################################
        
    def setFixation(self, x, y):
        self.coeff_layers, self.index_layers, self.scalingFactor = loadPickle(self.layersFile)
        self.fixation = np.array([x,y], dtype='int32')
        layer_shape = np.array(self.coeff_layers[0].shape, dtype='int32')
        img_x1, img_y1, img_x2, img_y2, ret_x1, ret_y1, ret_x2, ret_y2 = get_bounds(self.input_resolution, layer_shape, self.fixation)
        self.crop_coords = (img_x1, img_y1, img_x2, img_y2)
        for i in range(len(self.coeff_layers)):
            self.coeff_layers[i] = self.coeff_layers[i][ret_y1:ret_y2, ret_x1:ret_x2].ravel()
            self.index_layers[i] = self.index_layers[i][ret_y1:ret_y2, ret_x1:ret_x2].ravel()
        self.size = np.array([img_y2 - img_y1,img_x2 - img_x1], dtype='int32')
        self.createNormalizationImage()

    ########################################
    ############## GRAY SCALE ##############
    ########################################
            
    def sample_gray(self, img):
        self.sampledVector = np.zeros(numFields, dtype=numpy_types[types['RESULTS']])
        x1, y1, x2, y2 = self.crop_coords
        img = img[y1:y2, x1:x2].ravel()
        for i in range(len(self.coeff_layers)):
            sample(img, self.coeff_layers[i], self.index_layers[i], self.sampledVector)
            
    def createNormalizationImage_gray(self):
        self.backProjectedVector = np.zeros(self.size[0]*self.size[1], dtype=numpy_types[types['BAKC_PROJECTED']])
        ones = np.ones(self.input_resolution, dtype=numpy_types[types['INPUT']])
        self.sample(ones)
        for i in range(len(self.coeff_layers)):
            backProject(self.sampledVector, self.coeff_layers[i], self.index_layers[i], self.backProjectedVector)
        self.normalizationVector = np.copy(self.backProjectedVector)
        self.normalizationVector[np.where(self.normalizationVector==0)]=1
            
    def backProject_gray(self):
        self.backProjectedVector = np.zeros(self.size[0]*self.size[1], dtype=numpy_types[types['BAKC_PROJECTED']])
        for i in range(len(self.coeff_layers)):
            backProject(self.sampledVector, self.coeff_layers[i], self.index_layers[i], self.backProjectedVector)
        normalize(self.backProjectedVector, self.normalizationVector)
        
    #########################################
    ################## RGB ##################
    #########################################
        
    def sample_rgb(self, img):
        self.sampledVector = np.zeros((3,numFields), dtype=numpy_types[types['RESULTS']])
        x1, y1, x2, y2 = self.crop_coords
        img = img[y1:y2, x1:x2, :]
        R = img[:,:,0].ravel()
        G = img[:,:,1].ravel()
        B = img[:,:,2].ravel()
        for i in range(len(self.coeff_layers)):
            sampleRGB(R, G, B, self.coeff_layers[i], self.index_layers[i], self.sampledVector[0], self.sampledVector[1], self.sampledVector[2])
    
    def createNormalizationImage_rgb(self):
        self.backProjectedVector = np.zeros((3, self.size[0]*self.size[1]), dtype=numpy_types[types['BAKC_PROJECTED']])
        ones = np.ones((self.input_resolution[0], self.input_resolution[1], 3), dtype=numpy_types[types['INPUT']])
        self.sample(ones)
        for i in range(len(self.coeff_layers)):
            backProject(self.sampledVector[0], self.coeff_layers[i], self.index_layers[i], self.backProjectedVector[0])
        self.normalizationVector = np.copy(self.backProjectedVector[0])
        self.normalizationVector[np.where(self.normalizationVector==0)]=1
    
    def backProject_rgb(self):
        self.backProjectedVector = np.zeros((3, self.size[0]*self.size[1]), dtype=numpy_types[types['BAKC_PROJECTED']])
        for i in range(len(self.coeff_layers)):
            backProjectRGB(self.sampledVector[0], self.sampledVector[1], self.sampledVector[2], self.coeff_layers[i], self.index_layers[i], self.backProjectedVector[0], self.backProjectedVector[1], self.backProjectedVector[2])
        normalizeRGB(self.backProjectedVector[0], self.backProjectedVector[1], self.backProjectedVector[2], self.normalizationVector)

    #########################################
    ######## CALIBRATE SIZE AND TYPE ########
    #########################################

    def calibrate(self, img):
        rgb = (len(img.shape)==3) and (img.shape[-1]==3)
        if rgb:
            self.sampledVector = np.zeros((3,numFields), dtype=numpy_types[types['RESULTS']])
            self.sample = self.sample_rgb
            self.backProject = self.backProject_rgb
            self.createNormalizationImage = self.createNormalizationImage_rgb
            self.getResult = lambda: divideRGB(np.copy(self.sampledVector), self.scalingFactor)
        else:
            self.sampledVector = np.zeros(numFields, dtype=numpy_types[types['RESULTS']])
            self.sample = self.sample_gray
            self.backProject = self.backProject_gray
            self.createNormalizationImage = self.createNormalizationImage_gray
            self.getResult = lambda: (self.sampledVector/self.scalingFactor).astype(numpy_types[types['RESULTS']])

        self.setInputResolution(img.shape[1],img.shape[0])
        self.setFixation(img.shape[1]/2,img.shape[0]/2)

#===============================================================================================================================
#===============================================================================================================================
#                         ██╗  ██╗███████╗███╗   ███╗██╗███████╗██████╗ ██╗  ██╗███████╗██████╗ ███████╗
#                         ██║  ██║██╔════╝████╗ ████║██║██╔════╝██╔══██╗██║  ██║██╔════╝██╔══██╗██╔════╝
#                         ███████║█████╗  ██╔████╔██║██║███████╗██████╔╝███████║█████╗  ██████╔╝█████╗  
#                         ██╔══██║██╔══╝  ██║╚██╔╝██║██║╚════██║██╔═══╝ ██╔══██║██╔══╝  ██╔══██╗██╔══╝  
#                         ██║  ██║███████╗██║ ╚═╝ ██║██║███████║██║     ██║  ██║███████╗██║  ██║███████╗
#                         ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝╚═╝╚══════╝╚═╝     ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝
#===============================================================================================================================
#===============================================================================================================================
class Hemisphere:
    def __init__(self, fname):

        self.size = None
        self.coeff_layers = None
        self.index_layers = None
        self.scalingFactor = None
        self.backProjectedVector = None
        self.normalizationVector = None
        
        try:
            self.loadLayers(fname)
        except:
            print("Could not find previously saved layers, Please make sure to initlaize them!")

    def loadLayers(self, fname):
        self.coeff_layers, self.index_layers, self.scalingFactor = loadPickle(fname)
        self.size = np.array(self.coeff_layers[0].shape, dtype='int32')

        
    def createNormalizationImage(self):
        ones = np.ones(numFields, dtype=numpy_types[types['RESULTS']])
        for i in range(len(self.coeff_layers)):
            backProject(ones,  self.coeff_layers[i],  self.index_layers[i],  self.backProjectedVector)
        self.normalizationVector = np.copy(self.backProjectedVector)
        self.normalizationVector[np.where(self.normalizationVector==0)]=1
        
    def backProject_gray(self, sampledVector):
        self.backProjectedVector = np.zeros(self.size[0]*self.size[1], dtype=numpy_types[types['BAKC_PROJECTED']])
        for i in range(len(self.coeff_layers)):
            backProject(sampledVector,  self.coeff_layers[i],  self.index_layers[i],  self.backProjectedVector)
        normalize(self.backProjectedVector, self.normalizationVector)
        
    def backProject_rgb(self, sampledVector):
        self.backProjectedVector = np.zeros((3, self.size[0]*self.size[1]), dtype=numpy_types[types['BAKC_PROJECTED']])
        for i in range(len(self.coeff_layers)):
            backProjectRGB(sampledVector[0], sampledVector[1], sampledVector[2], self.coeff_layers[i], self.index_layers[i], self.backProjectedVector[0], self.backProjectedVector[1], self.backProjectedVector[2])
        normalize(self.backProjectedVector[0], self.normalizationVector)
        normalize(self.backProjectedVector[1], self.normalizationVector)
        normalize(self.backProjectedVector[2], self.normalizationVector)

#===============================================================================================================================
#===============================================================================================================================
#                                      ██████╗ ██████╗ ██████╗ ████████╗███████╗██╗  ██╗
#                                     ██╔════╝██╔═══██╗██╔══██╗╚══██╔══╝██╔════╝╚██╗██╔╝
#                                     ██║     ██║   ██║██████╔╝   ██║   █████╗   ╚███╔╝ 
#                                     ██║     ██║   ██║██╔══██╗   ██║   ██╔══╝   ██╔██╗ 
#                                     ╚██████╗╚██████╔╝██║  ██║   ██║   ███████╗██╔╝ ██╗
#                                      ╚═════╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝
#===============================================================================================================================
#===============================================================================================================================
class Cortex:
    def __init__(self, leftPath, rightPath):
        self.backProject = None
        
        self.left_hemi = Hemisphere(leftPath)
        for i in range(len(self.left_hemi.coeff_layers)):
            self.left_hemi.coeff_layers[i] = np.rot90(self.left_hemi.coeff_layers[i],1).ravel()
            self.left_hemi.index_layers[i] = np.rot90(self.left_hemi.index_layers[i],1).ravel()
        self.left_hemi.size = self.left_hemi.size[::-1]

        self.right_hemi = Hemisphere(rightPath)
        for i in range(len(self.right_hemi.coeff_layers)):
            self.right_hemi.coeff_layers[i] = np.rot90(self.right_hemi.coeff_layers[i],-1).ravel()
            self.right_hemi.index_layers[i] = np.rot90(self.right_hemi.index_layers[i],-1).ravel()
        self.right_hemi.size = self.right_hemi.size[::-1]
        
        self.size = np.array((self.left_hemi.size[0],self.left_hemi.size[1]*2), dtype='int32')
        
    def backProject_gray(self, sampledVector):
        self.left_hemi.backProject_gray(sampledVector)
        self.right_hemi.backProject_gray(sampledVector)
        
    def backProject_rgb(self, sampledVector):
        self.left_hemi.backProject_rgb(sampledVector)
        self.right_hemi.backProject_rgb(sampledVector)
        
    def calibrate(self, sampledVector):
        size = self.left_hemi.size
        numElements = size[0]*size[1]
        
        self.left_hemi.backProjectedVector = np.zeros(numElements, dtype=numpy_types[types['BAKC_PROJECTED']])
        self.right_hemi.backProjectedVector = np.zeros(numElements, dtype=numpy_types[types['BAKC_PROJECTED']])
        self.left_hemi.createNormalizationImage()
        self.right_hemi.createNormalizationImage()
        
        rgb = (len(sampledVector.shape)==2) and (sampledVector.shape[0]==3)
        if rgb:
            self.left_hemi.backProjectedVector = np.zeros((3,numElements), dtype=numpy_types[types['BAKC_PROJECTED']])
            self.left_hemi.backProjectedVector = np.zeros((3, numElements), dtype=numpy_types[types['BAKC_PROJECTED']])
            self.backProject = self.backProject_rgb
        else:
            self.backProject = self.backProject_gray

#===============================================================================================================================
#===============================================================================================================================
#                          ██████╗ ███████╗███╗   ██╗███████╗██████╗  █████╗ ████████╗ ██████╗ ██████╗ 
#                         ██╔════╝ ██╔════╝████╗  ██║██╔════╝██╔══██╗██╔══██╗╚══██╔══╝██╔═══██╗██╔══██╗
#                         ██║  ███╗█████╗  ██╔██╗ ██║█████╗  ██████╔╝███████║   ██║   ██║   ██║██████╔╝
#                         ██║   ██║██╔══╝  ██║╚██╗██║██╔══╝  ██╔══██╗██╔══██║   ██║   ██║   ██║██╔══██╗
#                         ╚██████╔╝███████╗██║ ╚████║███████╗██║  ██║██║  ██║   ██║   ╚██████╔╝██║  ██║
#                          ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝
#===============================================================================================================================
#===============================================================================================================================
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
                    self.locs = pickle.load(f, encoding=enc)
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
                    self.coeffs = np.squeeze(pickle.load(f, encoding=enc))
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
                a, b, c = coord
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
        savePickle(outputFile, (self.coeff_layers,self.index_layers,self.scalingFactor))
        config = {'types' : types,
                  'numpy_types' : numpy_types,
                  'numFields' : numFields}
        savePickle(configFile, config)