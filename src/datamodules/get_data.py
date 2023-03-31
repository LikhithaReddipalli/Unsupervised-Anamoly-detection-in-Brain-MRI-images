from torch.utils.data import Dataset
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
import skimage.transform as transform
from PIL import Image
import pandas as pd
import os
import math
import SimpleITK as sitk
import random 
from tqdm import tqdm
import pandas
"""
Everything related to the data sets
Parts of the Dataset classes and functions are taken from https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/datasets.py
"""


class MN_IXI(Dataset):  # Load combined Mixednormals and IXI data set for Training and validation

    def __init__(self, cfg, indices, stage):
        super(MN_IXI, self).__init__()
        'Initialization'
        self.stage = stage
        self.ID = []
        self.IDs_list = []
        self.preLoadList = []
        self.preloadListMask = []
        self.preloadListSeg = []
        self.preloadListOrig = []
        self.padParamsList = []
        self.erasingParamsList = []
        self.preloadListErased = []
        self.IDs = indices  # IDs of the Dataset (strings)
        self.cfg = cfg  # see config.py
        self.pad = cfg['pad']  # Pad data to the size of the largest scan
        self.imgSize = cfg['imageDim']  # image dimensions
        self.cropMode = cfg['cropMode']  # How to resize the image
        self.flow = cfg['curvatureFlow']  # filter for smoothing the images
        self.percentile = cfg['percentile']  # clipping
        self.preLoad = cfg['preLoad']
        self.spatialDims = cfg['spatialDims']  # 2D or 3D
        self.num_slices = cfg['imageDim'][2]  # "Depth" of the Volume
        self.brightnessRange = cfg['brightnessRange']  # augmentation
        self.contrastRange = cfg['contrastRange']  # augmentation
        self.randomContrast = cfg['randomBrightness']  # augmentation
        self.randomBrightness = cfg['randomContrast']  # augmentation
        self.rescaleFactor = cfg['rescaleFactor']
        cfg.IDs = self.IDs
        # This is important for multiprocessing with sitk and pytorch
        sitk.ProcessObject.SetGlobalDefaultThreader("Platform")

        cfg.dataDir = cfg.path.MN_IXI.dataDir

        if self.stage == "val":
            df = pandas.read_csv(cfg.path.pathBase + '/splits/combined_val.csv')
        else:
            df = pandas.read_csv(cfg.path.pathBase + '/splits/combined_train.csv')
        self.age = df[['img_name', 'age']]

        self.ID_key = 'img_name'
        self.replace_str = '_t1.nii.gz'
        cfg.DataSet = 'MN_IXI'
        self.DataSet = 'MN_IXI'
        # Preload Data to RAM
        if self.preLoad:
            print('preloading Data')
            for index in tqdm(range(len(self.IDs))):
                vol, pad_params = procces(cfg, index)
                # Fill preloadlists
                self.preLoadList.append(vol)
                self.padParamsList.append(pad_params)

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        sample = {}

        if self.spatialDims == '2D':
            self.cropInt = random.randint(0, self.num_slices) # for 2D we only sample a random slice of the volume

        if self.preLoad:
            vol = self.preLoadList[index]
        else:
            vol, pad_params = procces(self.cfg, index)

        self.ID = self.IDs[index]
        # Get Age information
        ''' 
        As IXI and Mixed normals are combined to same folder, using IXI string in ID to replace the extensions
        '''
        if self.ID[0:3] == 'IXI':
            age = self.age[self.age[self.ID_key] == self.ID.replace(self.replace_str,'_t1.nii.gz')]
        else:
            age = self.age[self.age[self.ID_key] == self.ID.replace(self.replace_str, '')]

        sample['Age'] = age.age.item()
        sample['ID'] = self.ID
        vol = torch.from_numpy(vol)

        if self.spatialDims == '2D':
            vol = vol[self.cropInt, :, :]

        sample['vol'] = vol
        sample['stage'] = self.stage
        sample['Dataset'] = self.DataSet
        return sample



class MN_IXI_eval(Dataset):  # Load Mixednormals and IXI data set for Evaluation

    def __init__(self, cfg, indices, stage):
        super(MN_IXI_eval, self).__init__()
        'Initialization'
        self.stage = stage
        self.ID = []
        self.IDs_list = []
        self.preLoadList = []
        self.preloadListMask = []
        self.preloadListSeg = []
        self.preloadListOrig = []
        self.padParamsList = []
        self.erasingParamsList = []
        self.preloadListErased = []
        self.IDs = indices  # IDs of the Dataset (strings)
        self.cfg = cfg  # see config.py
        self.pad = cfg['pad']  # Pad data to the size of the largest scan
        self.imgSize = cfg['imageDim']  # image dimensions
        self.cropMode = cfg['cropMode']  # How to resize the image
        self.flow = cfg['curvatureFlow']  # filter for smoothing the images
        self.percentile = cfg['percentile']  # clipping
        self.preLoad = cfg['preLoad']
        self.spatialDims = cfg['spatialDims']  # 2D or 3D
        self.num_slices = cfg['imageDim'][2]  # "Depth" of the Volume
        self.brightnessRange = cfg['brightnessRange']  # augmentation
        self.contrastRange = cfg['contrastRange']  # augmentation
        self.randomContrast = cfg['randomBrightness']  # augmentation
        self.randomBrightness = cfg['randomContrast']  # augmentation
        self.rescaleFactor = cfg['rescaleFactor']
        self.resizedEvaluation = cfg.resizedEvaluation
        cfg.IDs = self.IDs
        # This is important for multiprocessing with sitk and pytorch
        sitk.ProcessObject.SetGlobalDefaultThreader("Platform")

        cfg.dataDir = cfg.path.MN_IXI.dataDir
        self.dataDir = cfg.dataDir
        if self.stage == "val":
            df = pandas.read_csv(cfg.path.pathBase + '/splits/combined_val.csv')
        else:
            df = pandas.read_csv(cfg.path.pathBase + '/splits/combined_test.csv')
        self.age = df[['img_name', 'age']]
        self.ID_key = 'img_name'
        self.replace_str = '_t1.nii.gz'
        cfg.DataSet = 'MN_IXI'
        self.DataSet = 'MN_IXI'

        # Preload Data to RAM
        if self.preLoad:
            print('preloading Data')
            for index in tqdm(range(len(self.IDs))):
                vol, pad_params = procces(cfg, index)
                # Fill preloadlists
                self.preLoadList.append(vol)
                self.padParamsList.append(pad_params)

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        sample = {}

        if self.spatialDims == '2D':  # for 2D we only sample a random slice of the volume
            self.cropInt = random.randint(0, self.num_slices )


        if self.preLoad:
            vol = self.preLoadList[index]
            pad_params = self.padParamsList[index]
        else:
            vol, pad_params = procces(self.cfg, index)

        self.ID = self.IDs[index]
        # Get Age information
        if self.ID[0:3] == 'IXI':
            age = self.age[self.age[self.ID_key] == self.ID.replace(self.replace_str,'_t1.nii.gz')]
        else:
            age = self.age[self.age[self.ID_key] == self.ID.replace(self.replace_str, '')]
        sample['age'] = age.age.item()
        sample['ID'] = self.ID
        vol = torch.from_numpy(vol)

        if self.spatialDims == '2D':
            if self.cfg.get('showAllSlices',False):
                vol = vol
            else :
                vol = vol[self.cropInt,:,:]

        # Load input with original size - This is only required for evaluation

        if self.ID[0] == 'I':
             self.dataDir = '/data/Reddipalli/IXI/t1/'
        else:
             self.dataDir = '/data/Reddipalli/MixedNormals/t1/'

        path_image_nii = self.dataDir + self.ID
        data_orig_nii = sitk.ReadImage(path_image_nii, sitk.sitkFloat32)
        data_orig_nii = sitk.CurvatureFlow(image1=data_orig_nii, timeStep=0.125, numberOfIterations=3)
        data_orig_nii32 = sitk.Cast(data_orig_nii, sitk.sitkFloat32)
        data_orig = sitk.GetArrayFromImage(data_orig_nii32)
        # Get Segmentation ground truth
        data_seg = np.zeros_like(data_orig)  # only healthy samples
        # Load Mask
        mask_path = self.dataDir.replace('t1', 'mask') + self.ID.replace('_t1.nii.gz',
                                                                             '_mask.nii.gz')
        mask_nii = sitk.ReadImage(mask_path, sitk.sitkFloat32)
        data_mask = sitk.GetArrayFromImage(mask_nii)
        data_orig *= data_mask

        if data_orig.mean() < 0:  # for negative ranges.
            data_orig[data_mask > 0] -= np.amin(data_orig[
                                                    data_mask > 0])  # To be sure, even if the normalization should handle this. Sometimes there are ranges from - to +

        if self.pad:
            dim1, dim2, dim3 = 158, 190, 140
            data_orig, _ = pad_data(data_orig, dim1, dim2, dim3)
            data_mask, _ = pad_data(data_mask, dim1, dim2, dim3)
            data_seg, _ = pad_data(data_seg, dim1, dim2, dim3)

        data_orig[np.isnan(data_orig)] = 0  # eliminate NaNs

        # Clip Data
        if self.percentile:  # Get values
            low = np.percentile(data_orig[data_mask > 0], 1)
            high = np.percentile(data_orig[data_mask > 0], 99)
            data_orig[data_orig < low] = low
            data_orig[data_orig > high] = high

        if self.resizedEvaluation:
            if self.cropMode == 'cube':
                data_orig = transform.resize(data_orig, (self.num_slices, self.imgSize[0], self.imgSize[1]),
                                             anti_aliasing=True, order=3)
                data_mask = transform.resize(data_mask, (self.num_slices, self.imgSize[0], self.imgSize[1]),
                                             anti_aliasing=True, order=3)
                data_seg = transform.resize(data_seg, (self.num_slices, self.imgSize[0], self.imgSize[1]),
                                            anti_aliasing=True, order=3)
            elif self.cropMode == 'isotropic':

                data_orig = transform.rescale(data_orig, self.rescaleFactor, anti_aliasing=True, order=3)
                data_mask = transform.rescale(data_mask, self.rescaleFactor, anti_aliasing=True, order=3)
                data_seg = transform.rescale(data_seg, self.rescaleFactor, anti_aliasing=True, order=3)
            data_seg[data_seg < 0.01] = 0
            data_mask = data_mask > 0.1  # this ensures values inside the brain.

        sample['vol_orig'], sample['seg_orig'], sample['mask_orig'] = data_orig,  data_seg,  data_mask
        sample['stage'] = self.stage
        sample['vol'] = vol
        sample['pad_params'] = pad_params
        sample['Dataset'] = self.DataSet
        return sample



class Stroke(Dataset):  # Load Stroke/ATLAS data set for Evaluation (pixel-, slice- and sample-wise)

    def __init__(self, cfg, indices, stage):
        super(Stroke, self).__init__()
        'Initialization'
        self.stage = stage
        self.ID = []
        self.IDs_list = []
        self.preLoadList = []
        self.preloadListMask = []
        self.preloadListSeg = []
        self.preloadListOrig = []
        self.padParamsList = []
        self.erasingParamsList = []
        self.preloadListErased = []
        self.IDs = indices  # IDs of the Dataset (strings)
        self.cfg = cfg  # see config.py
        self.pad = cfg['pad']  # Pad data to the size of the largest scan
        self.imgSize = cfg['imageDim']  # image dimensions
        self.cropMode = cfg['cropMode']  # How to resize the image
        self.flow = cfg['curvatureFlow']  # filter for smoothing the images
        self.percentile = cfg['percentile']  # clipping
        self.preLoad = cfg['preLoad']
        self.spatialDims = cfg['spatialDims']  # 2D or 3D
        self.num_slices = cfg['imageDim'][2]  # "Depth" of the Volume
        self.brightnessRange = cfg['brightnessRange']  # augmentation
        self.contrastRange = cfg['contrastRange']  # augmentation
        self.randomContrast = cfg['randomBrightness']  # augmentation
        self.randomBrightness = cfg['randomContrast']  # augmentation
        self.rescaleFactor = cfg['rescaleFactor']
        self.resizedEvaluation = cfg.resizedEvaluation
        cfg.IDs = self.IDs
        # This is important for multiprocessing with sitk and pytorch
        sitk.ProcessObject.SetGlobalDefaultThreader("Platform")

        cfg.dataDir = cfg.path.Stroke.dataDir
        self.dataDir = cfg.dataDir
        if self.stage == "val":
            df = pandas.read_csv(cfg.path.pathBase + '/splits/Stroke_val.csv')
        else:
            df = pandas.read_csv(cfg.path.pathBase + '/splits/Stroke_test.csv')
        if 'c0004s0003t01_t1.nii.gz' in self.IDs:
            self.IDs.remove('c0004s0003t01_t1.nii.gz')
        self.age = df[['img_name']]  # no age in Stroke
        self.ID_key = 'img_name'
        self.replace_str = '_t1.nii.gz'
        cfg.DataSet = 'Stroke'
        self.DataSet = 'Stroke'
        # Preload Data to RAM

        cfg.IDs = self.IDs
        if self.preLoad:
            print('preloading Data')
            for index in tqdm(range(len(self.IDs))):
                vol, pad_params = procces(cfg, index)
                # Fill preloadlists
                self.preLoadList.append(vol)
                self.padParamsList.append(pad_params)

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        sample = {}

        if self.spatialDims == '2D':    # for 2D we only sample a random slice of the volume
            self.cropInt = random.randint(0, self.num_slices)

        if self.preLoad:
            vol = self.preLoadList[index]
            pad_params = self.padParamsList[index]
        else:
            vol, pad_params = procces(self.cfg, index)

        self.ID = self.IDs[index]

        # Get Age information
        sample['age'] = np.NaN  # no age information in Stroke Set
        sample['ID'] = self.ID
        vol = torch.from_numpy(vol)

        if self.spatialDims == '2D':
            if self.cfg.get('showAllSlices',False):
                vol = vol
            else :
                vol = vol[self.cropInt,:,:]

        sample['vol'] = vol
        sample['pad_params'] = pad_params

        # Load input with original size
        path_image_nii = self.dataDir + self.ID
        data_orig_nii = sitk.ReadImage(path_image_nii, sitk.sitkFloat32)
        data_orig_nii = sitk.CurvatureFlow(image1=data_orig_nii, timeStep=0.125, numberOfIterations=3)
        data_orig_nii32 = sitk.Cast(data_orig_nii, sitk.sitkFloat32)
        data_orig = sitk.GetArrayFromImage(data_orig_nii32)
        # Get Segmentation
        if os.path.isfile(path_image_nii.replace('t1', 'seg')):  # if we have a segmentation mask
            segmentation_path = path_image_nii.replace('t1', 'seg')
            seg_nii = sitk.ReadImage(segmentation_path, sitk.sitkFloat32)
            data_seg = sitk.GetArrayFromImage(seg_nii)
            data_seg[data_seg > 0] = 1
        else:
            data_seg = np.zeros_like(data_orig)
            # Load Mask
        mask_path = path_image_nii.replace('t1', 'mask')
        mask_nii = sitk.ReadImage(mask_path, sitk.sitkFloat32)
        data_mask = sitk.GetArrayFromImage(mask_nii)
        data_orig *= data_mask

        if data_orig.mean() < 0:  # for negative ranges.
            data_orig[data_mask > 0] -= np.amin(data_orig[
                                                    data_mask > 0])  # To be sure, even if the normalization should handle this. Sometimes there are ranges from - to +

        if self.pad:
            dim1, dim2, dim3 = 158, 190, 140
            data_orig, _ = pad_data(data_orig, dim1, dim2, dim3)
            data_mask, _ = pad_data(data_mask, dim1, dim2, dim3)
            data_seg, _ = pad_data(data_seg, dim1, dim2, dim3)

        data_orig[np.isnan(data_orig)] = 0  # eliminate NaNs

        # Clip Data
        if self.percentile:  # Get values
            low = np.percentile(data_orig[data_mask > 0], 1)
            high = np.percentile(data_orig[data_mask > 0], 99)
            data_orig[data_orig < low] = low
            data_orig[data_orig > high] = high

        if self.resizedEvaluation:
            if self.cropMode == 'cube':
                data_orig = transform.resize(data_orig, (self.num_slices, self.imgSize[0], self.imgSize[1]),
                                             anti_aliasing=True, order=3)
                data_mask = transform.resize(data_mask, (self.num_slices, self.imgSize[0], self.imgSize[1]),
                                             anti_aliasing=True, order=3)
                data_seg = transform.resize(data_seg, (self.num_slices, self.imgSize[0], self.imgSize[1]),
                                            anti_aliasing=True, order=3)
            elif self.cropMode == 'isotropic':
                data_orig = transform.rescale(data_orig, self.rescaleFactor, anti_aliasing=True, order=3)
                data_mask = transform.rescale(data_mask, self.rescaleFactor, anti_aliasing=True, order=3)
                data_seg = transform.rescale(data_seg, self.rescaleFactor, anti_aliasing=True, order=3)
            data_seg[data_seg < 0.01] = 0
            data_mask = data_mask > 0.1  # this ensures values inside the brain.
        data_orig = scale_data(data_orig, data_mask, self.cfg)

        sample['vol_orig'], sample['seg_orig'], sample['mask_orig'] = data_orig, data_seg,data_mask
        sample['stage'] = self.stage
        sample['Dataset'] = self.DataSet
        return sample


class Brats19(Dataset):# Load Brats19 data set for Evaluation (pixel-, slice- and sample-wise)
   
    def __init__(self, cfg, indices, stage):
        super(Brats19, self).__init__()
        'Initialization'
        self.stage = stage
        self.ID = [] 
        self.IDs_list = []
        self.preLoadList = []
        self.preloadListMask = []
        self.preloadListSeg = []
        self.preloadListOrig = []
        self.padParamsList = []
        self.erasingParamsList = []
        self.preloadListErased = []
        self.IDs = indices # IDs of the Dataset (strings)
        self.cfg = cfg # see config.py
        self.pad = cfg['pad'] # Pad data to the size of the largest scan
        self.imgSize = cfg['imageDim'] # image dimensions
        self.cropMode = cfg['cropMode'] # How to resize the image 
        self.flow = cfg['curvatureFlow'] # filter for smoothing the images
        self.percentile = cfg['percentile'] # clipping
        self.preLoad = cfg['preLoad'] 
        self.spatialDims = cfg['spatialDims'] # 2D or 3D
        self.num_slices = cfg['imageDim'][2] # "Depth" of the Volume
        self.brightnessRange = cfg['brightnessRange'] # augmentation
        self.contrastRange = cfg['contrastRange']# augmentation
        self.randomContrast = cfg['randomBrightness']# augmentation
        self.randomBrightness = cfg['randomContrast']# augmentation
        self.rescaleFactor = cfg['rescaleFactor']
        self.resizedEvaluation = cfg.resizedEvaluation
        cfg.IDs = self.IDs
        # This is important for multiprocessing with sitk and pytorch
        sitk.ProcessObject.SetGlobalDefaultThreader("Platform")

        cfg.dataDir = cfg.path.Brats19.dataDir
        self.dataDir = cfg.dataDir
        if self.stage == "val":
            df = pandas.read_csv(cfg.path.pathBase + '/splits/Brats19_val.csv')
        else:
            df = pandas.read_csv(cfg.path.pathBase + '/splits/Brats19_test.csv')
        self.age = df[['img_name','age']]
        self.ID_key = 'img_name'
        self.replace_str = '_t1.nii.gz'
        cfg.DataSet = 'Brats19'
        self.DataSet = 'Brats19'
        # Preload Data to RAM
        if self.preLoad : 
            print('preloading Data')
            for index in tqdm(range(len(self.IDs))) :
                vol, pad_params = procces(cfg,index)
                # Fill preloadlists
                self.preLoadList.append(vol)
                self.padParamsList.append(pad_params) 

    
    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        sample = {}

        if self.spatialDims == '2D':
            self.cropInt = random.randint(0, self.num_slices)  # for 2D we only sample a random slice of the volume

        if self.preLoad:
            vol = self.preLoadList[index]
            pad_params = self.padParamsList[index]
        else:
            vol, pad_params = procces(self.cfg, index)

        self.ID = self.IDs[index]
        # Get Age information
        age = self.age[self.age[self.ID_key]==self.ID.replace(self.replace_str,'')]
        sample['age'] = age.age.item()
        sample['ID'] = self.ID
        vol = torch.from_numpy(vol)

        if self.spatialDims == '2D':
            if self.cfg.get('showAllSlices',False):
                vol = vol
            else :
                vol = vol[self.cropInt,:,:]

        sample['vol'] = vol
        sample['pad_params'] = pad_params

        # Load input with original size - This is only required for evaluation --> TBD: also use Process() here with 
        path_image_nii = self.dataDir + self.ID
        data_orig_nii = sitk.ReadImage(path_image_nii,sitk.sitkFloat32)
        data_orig_nii = sitk.CurvatureFlow(image1 = data_orig_nii, timeStep = 0.125, numberOfIterations = 3)
        data_orig_nii32 = sitk.Cast(data_orig_nii,sitk.sitkFloat32)
        data_orig = sitk.GetArrayFromImage(data_orig_nii32)         
        # Get Segmentation
        if os.path.isfile(path_image_nii.replace('t1','seg')): # if we have a segmentation mask
            segmentation_path = path_image_nii.replace('t1','seg') 
            seg_nii = sitk.ReadImage(segmentation_path,sitk.sitkFloat32)
            data_seg = sitk.GetArrayFromImage(seg_nii)
            data_seg[data_seg>0] = 1
        else:
                data_seg = np.zeros_like(data_orig) 
        # Load Mask
        mask_path = path_image_nii.replace('t1','mask')
        mask_nii = sitk.ReadImage(mask_path,sitk.sitkFloat32)
        data_mask = sitk.GetArrayFromImage(mask_nii)
        data_orig *= data_mask

        if data_orig.mean() < 0: # for negative ranges. 
            data_orig[data_mask>0] -= np.amin(data_orig[data_mask>0]) # To be sure, even if the normalization should handle this. Sometimes there are ranges from - to +
        
        if self.pad :
            dim1, dim2, dim3 = 158, 190, 140
            data_orig, _ = pad_data(data_orig,dim1,dim2,dim3)
            data_mask, _ = pad_data(data_mask,dim1,dim2,dim3)
            data_seg, _ = pad_data(data_seg,dim1,dim2,dim3)

        data_orig[np.isnan(data_orig)] = 0 # eliminate NaNs

        # Clip Data
        if self.percentile : # Get values
            low = np.percentile(data_orig[data_mask>0], 1)
            high = np.percentile(data_orig[data_mask>0], 99)
            data_orig[data_orig < low] = low
            data_orig[data_orig > high] = high

        if self.resizedEvaluation : 
            if self.cropMode == 'cube':
                data_orig = transform.resize(data_orig,(self.num_slices,self.imgSize[0],self.imgSize[1]),anti_aliasing = True,order=3)
                data_mask = transform.resize(data_mask,(self.num_slices,self.imgSize[0],self.imgSize[1]),anti_aliasing = True,order=3)
                data_seg = transform.resize(data_seg,(self.num_slices,self.imgSize[0],self.imgSize[1]),anti_aliasing = True,order=3)
            elif self.cropMode == 'isotropic':
                data_orig = transform.rescale(data_orig,self.rescaleFactor,anti_aliasing = True,order=3)
                data_mask = transform.rescale(data_mask,self.rescaleFactor,anti_aliasing = True,order=3)
                data_seg = transform.rescale(data_seg,self.rescaleFactor,anti_aliasing = True,order=3)
            data_seg[data_seg<0.01] = 0
            data_mask = data_mask > 0.1 # this ensures values inside the brain. 
        data_orig = scale_data(data_orig,data_mask,self.cfg)

        sample['vol_orig'], sample['seg_orig'], sample['mask_orig'] = data_orig, data_seg, data_mask
        sample['stage'] = self.stage
        sample['Dataset'] = self.DataSet
        return sample


     ### Helper functions ###
def normalize(img, maxval, reshape=False):
    """Scales images to be roughly [-1024 1024]."""
    
    if img.max() > maxval:
        raise Exception("max image value ({}) higher than expected bound ({}).".format(img.max(), maxval))
    
    img = (2 * (img.astype(np.float32) / maxval) - 1.) * 1024

    if reshape:
        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # add color channel
        img = img[None, :, :] 
    
    return img


def procces(cfg,index): # everything that can be done for preloading
            # Load Image/Volume
            ID = cfg.IDs[index]

            image_nii = sitk.ReadImage(cfg.dataDir + ID, sitk.sitkFloat32)
            image_nii = sitk.CurvatureFlow(image1 = image_nii, timeStep = 0.125, numberOfIterations = 3)
            image_nii32 = sitk.Cast(image_nii,sitk.sitkFloat32) # convert to float32
            vol = sitk.GetArrayFromImage(image_nii32)

            # Load Mask for training
            if cfg.DataSet == 'MN_IXI':
                path_mask = (cfg.dataDir.replace('t1','mask') + ID.replace('_t1.nii.gz','_mask.nii.gz'))
            else : 
                path_mask = (cfg.dataDir + ID).replace('t1','mask')
            mask_nii = sitk.ReadImage(path_mask, sitk.sitkFloat32) 
            mask = sitk.GetArrayFromImage(mask_nii)

            vol *= mask
            if vol.mean() < 0: # for negative ranges. 
                vol[mask>0] -= np.amin(vol[mask>0]) # To be sure, even if the normalization should handle this. Sometimes there are ranges from - to +
            
            #  pad Volume and mask
            if cfg.pad :
                dim1, dim2, dim3 = 158, 190, 140
                vol, pad_params = pad_data(vol,dim1,dim2,dim3)
                mask, _ = pad_data(mask,dim1,dim2,dim3)
            else :
                pad_params = [0,0,0,0,0,0]

            # Maintain Image/Volume
            vol[np.isnan(vol)] = 0 # Eliminate NaNs

            # Augment -> not used atm
            if cfg.randomBrightness :
                vol = augment_brightness_multiplicative(vol,cfg.brightnessRange)
            if cfg.randomContrast :
                vol = augment_contrast(vol,cfg.contrastRange)

            # Clipping
            if cfg.percentile : # Get values
                low = np.percentile(vol[mask>0], 1)
                high = np.percentile(vol[mask>0], 99)
                vol[vol < low] = low
                vol[vol > high] = high

            # Resize

            if cfg.cropMode == 'cube':
                if cfg.spatialDims == '2D':
                    cfg.num_slices = vol.squeeze().shape[0]  # dont change the depth
                    vol = transform.resize(vol, (cfg.num_slices, cfg.imageDim[0], cfg.imageDim[1]), anti_aliasing=True,
                                           order=3)
                    mask = transform.resize(mask, (cfg.num_slices, cfg.imageDim[0], cfg.imageDim[1]), anti_aliasing=True,
                                            order=3)


            elif cfg.cropMode == 'isotropic':

                vol = transform.rescale(vol,cfg.rescaleFactor,anti_aliasing = True,order=3)
                mask = transform.rescale(mask,cfg.rescaleFactor,anti_aliasing = True,order=3)

            mask = mask > 0.1 # this ensures values inside the brain. 

            vol = scale_data(vol,mask,cfg)
            return vol, pad_params

def pad_data(nii_sample_dil, targetdim0, targetdim1, targetdim2): # [targetdim2 (depth), targetdim0, targetdim1]


    nii_size = nii_sample_dil.shape

    if nii_size[1] < targetdim1:
        pad_width = (targetdim1 - nii_size[1]) / 2
        pad_left = math.ceil(pad_width)
        pad_right = math.floor(pad_width)
    elif nii_size[1] >= targetdim1:
        center = round(nii_size[1] / 2)
        pad_left = 0
        pad_right = 0
    if nii_size[2] < targetdim0:
        pad_height = (targetdim0 - nii_size[2]) / 2
        pad_top = math.ceil(pad_height)
        pad_bottom = math.floor(pad_height)
    elif nii_size[2] >= targetdim0:
        center = round(nii_size[2] / 2)
        pad_top = 0
        pad_bottom = 0
    if nii_size[0] < targetdim2:
        pad_height = (targetdim2 - nii_size[0]) / 2
        pad_front = math.ceil(pad_height)
        pad_back = math.floor(pad_height)
    elif nii_size[0] >= targetdim2:
        center = round(nii_size[0] / 2)
        pad_front = 0
        pad_back = 0
    pad_params = [pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back]
    nii_sample_dil = np.pad(nii_sample_dil, ((pad_front,pad_back),(pad_left, pad_right), (pad_top, pad_bottom)))
    #nii_sample_dil = transform.resize(nii_sample_dil, (140, 180, 160),anti_aliasing=True,order=3)
    return nii_sample_dil, pad_params

def augment_contrast(data_sample, contrast_range, preserve_range=True): # Zimmerer

    mn = data_sample.mean()
    if preserve_range:
        minm = data_sample.min()
        maxm = data_sample.max()
    if np.random.random() < 0.5 and contrast_range[0] < 1:
        factor = np.random.uniform(contrast_range[0], 1)
    else:
        factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])
    data_sample = (data_sample - mn) * factor + mn
    if preserve_range:
        data_sample[data_sample < minm] = minm
        data_sample[data_sample > maxm] = maxm
    return data_sample

def augment_brightness_multiplicative(data_sample, multiplier_range=(0.5, 2)): # Zimmerer
    multiplier = np.random.uniform(multiplier_range[0], multiplier_range[1])
    data_sample *= multiplier
    return data_sample

def scale_data(vol,mask,params):
    #Normalize -> Scale to [0,1]
    if params.get('standardize') == 'std':
        #Standardize
        mean_vol = np.mean(vol[mask>0])
        std_vol = np.std(vol[mask>0])
        vol = (vol -mean_vol)/(std_vol + 0.000001)
    else:
        # min max normalize
        vol -= np.amin(vol[mask>0])
        vol /= (np.amax(vol[mask>0]))

    return vol
