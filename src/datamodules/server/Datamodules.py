from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import os
from torchvision import datasets, transforms
from pytorch_lightning import LightningDataModule
from pytorch_lightning.core.lightning import LightningModule
import src.datamodules.get_data as get_data
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

class MN_IXI_combined(LightningDataModule):

    def __init__(self, cfg):
        super(MN_IXI_combined, self).__init__()
        self.cfg = cfg
        # load data paths and indices
        # Mixednormals and IXI
        path = self.cfg.path.MN_IXI.IDs
        # print(path)
        '''folds = range(len(cfg.path.MN.IDs.train))

        for fold in folds:'''

        df_train = pd.read_csv(path['train'])
        df_val = pd.read_csv(path['val'])

        cfg.MN_IXI_train_ids = df_train["img_path"].apply(os.path.basename).to_list()
        cfg.MN_IXI_val_ids = df_val["img_path"].apply(os.path.basename).to_list()

        """with open(path,'rb') as f :
            tempDict = pickle.load(f)
            cfg.MN_train_ids =  tempDict['TrainIDs']
            cfg.MN_val_ids =  tempDict['ValIDs']
            #print("cfg.MN_train_ids",cfg.MN_train_ids)
            #print("cfg.MN_train_ids1", cfg.MN_train_ids)
            cfg.MN_train_ids[0] = cfg.MN_train_ids[0][0:2]
            cfg.MN_val_ids[0] = cfg.MN_val_ids[0][0:2]"""

    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        if not hasattr(self, 'train_MN_IXI'):
            self.train_MN_IXI = get_data.MN_IXI(self.cfg, self.cfg.MN_IXI_train_ids, 'train')
            self.val_MN_IXI = get_data.MN_IXI(self.cfg, self.cfg.MN_IXI_val_ids, 'val')

    def train_dataloader(self):
        return DataLoader(self.train_MN_IXI, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers,
                          pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_MN_IXI, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers,
                          pin_memory=True, shuffle=False)

    def show_sample(self, index=None):  # show a random sample for the data set
        if not index:  # get a random index if not specified
            index = np.random.randint(0, len(self.indices))
        data = get_data.NIH_DataSet(indices=self.indices, datatable=self.data_table)
        sample = data.__getitem__(index)
        img, label = sample['img'], sample['label']
        plt.imshow(img, 'gray')
        plt.title('Class: {}'.format(label))
        plt.show()


class MixedNormals(LightningDataModule):

    def __init__(self, cfg):
        super(MixedNormals, self).__init__()
        self.cfg = cfg
        # load data paths and indices
        # Mixednormals
        path = self.cfg.path.MN.IDs
        # print(path)
        '''folds = range(len(cfg.path.MN.IDs.train))

        for fold in folds:'''

        df_train = pd.read_csv(path['train'][0])
        df_val = pd.read_csv(path['val'][0])

        cfg.MN_train_ids = df_train["img_path"].apply(os.path.basename)[0:2].to_list()
        cfg.MN_val_ids = df_val["img_path"].apply(os.path.basename)[0:2].to_list()

        """with open(path,'rb') as f :
            tempDict = pickle.load(f)
            cfg.MN_train_ids =  tempDict['TrainIDs']
            cfg.MN_val_ids =  tempDict['ValIDs']
            #print("cfg.MN_train_ids",cfg.MN_train_ids)
            #print("cfg.MN_train_ids1", cfg.MN_train_ids)
            cfg.MN_train_ids[0] = cfg.MN_train_ids[0][0:2]
            cfg.MN_val_ids[0] = cfg.MN_val_ids[0][0:2]"""

    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        if not hasattr(self, 'train_MN'):
            self.train_MN = get_data.MN(self.cfg, self.cfg.MN_train_ids, 'train')
            self.val_MN = get_data.MN(self.cfg, self.cfg.MN_val_ids, 'val')

    def train_dataloader(self):
        return DataLoader(self.train_MN, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers,
                          pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_MN, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers,
                          pin_memory=True, shuffle=False)

    def show_sample(self, index=None):  # show a random sample for the data set
        if not index:  # get a random index if not specified
            index = np.random.randint(0, len(self.indices))
        data = get_data.NIH_DataSet(indices=self.indices, datatable=self.data_table)
        sample = data.__getitem__(index)
        img, label = sample['img'], sample['lab-el']
        plt.imshow(img, 'gray')
        plt.title('Class: {}'.format(label))
        plt.show()


class ixiNormal(LightningDataModule):

    def __init__(self, cfg):
        super(ixiNormal, self).__init__()
        self.cfg = cfg
        # load data paths and indices
        # ixidata
        path = self.cfg.path.IXI.IDs
        with open(path, 'rb') as f:
            tempDict = pickle.load(f)
            cfg.IXI_train_ids = tempDict['TrainIDs']
            cfg.IXI_val_ids = tempDict['ValIDs']
            # print("cfg.MN_train_ids",cfg.MN_train_ids)
            # print("cfg.MN_train_ids1", cfg.MN_train_ids)
            cfg.IXI_train_ids[0] = cfg.IXI_train_ids[0][0:2]
            cfg.IXI_val_ids[0] = cfg.IXI_val_ids[0][0:2]

    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        if not hasattr(self, 'train_IXI'):
            self.train_IXI = get_data.MN(self.cfg, self.cfg.IXI_train_ids[0], 'train')
            self.val_IXI = get_data.MN(self.cfg, self.cfg.IXI_val_ids[0], 'val')

    def train_dataloader(self):
        return DataLoader(self.train_IXI, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers,
                          pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_IXI, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers,
                          pin_memory=True, shuffle=False)

    def show_sample(self, index=None):  # show a random sample for the data set
        if not index:  # get a random index if not specified
            index = np.random.randint(0, len(self.indices))
        data = get_data.NIH_DataSet(indices=self.indices, datatable=self.data_table)
        sample = data.__getitem__(index)
        img, label = sample['img'], sample['lab-el']
        plt.imshow(img, 'gray')
        plt.title('Class: {}'.format(label))
        plt.show()

class MN_IXI_eval(LightningDataModule):  # held out Set for Evaluation

    def __init__(self, cfg):
        super(MN_IXI_eval, self).__init__()
        self.cfg = cfg
        # load data paths and indices
        # Mixednormals nad ixi
        path = self.cfg.path.MN_IXI.IDs
        df_val = pd.read_csv(path['val'])
        df_test = pd.read_csv(path['test'])

        cfg.MN_IXI_val_ids = df_val["img_path"].apply(os.path.basename).to_list()
        cfg.MN_IXI_test_ids = df_test["img_path"].apply(os.path.basename).to_list()
        '''with open(path,'rb') as f :
            tempDict = pickle.load(f)
            cfg.MN_val_ids =  tempDict['ValIDs']
            cfg.MN_test_ids =  tempDict['TestIDs']
            cfg.MN_val_ids[0] = cfg.MN_val_ids[0][0:2]
            cfg.MN_test_ids[0] = cfg.MN_test_ids[0][0:2]'''

    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        if not hasattr(self, 'val_MN_IXI'):  # avoid unnecessary loading
            self.val_MN_IXI = get_data.MN_IXI_eval(self.cfg, self.cfg.MN_IXI_val_ids, 'val')
            self.test_MN_IXI = get_data.MN_IXI_eval(self.cfg, self.cfg.MN_IXI_test_ids, 'test')

    def val_dataloader(self):
        return DataLoader(self.val_MN_IXI, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_MN_IXI, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def show_sample(self, index=None):  # show a random sample for the data set
        if not index:  # get a random index if not specified
            index = np.random.randint(0, len(self.indices))
        data = get_data.NIH_DataSet(indices=self.indices, datatable=self.data_table)
        sample = data.__getitem__(index)
        img, label = sample['img'], sample['label']
        plt.imshow(img, 'gray')
        plt.title('Class: {}'.format(label))
        plt.show()

class MixedNormals_eval(LightningDataModule):  # held out Set for Evaluation

    def __init__(self, cfg):
        super(MixedNormals_eval, self).__init__()
        self.cfg = cfg
        # load data paths and indices
        # Mixednormals
        path = self.cfg.path.MN.IDs
        df_val = pd.read_csv(path['val'][0])
        df_test = pd.read_csv(path['test'])

        cfg.MN_val_ids = df_val["img_path"].apply(os.path.basename).to_list()
        cfg.MN_test_ids = df_test["img_path"].apply(os.path.basename).to_list()
        '''with open(path,'rb') as f :
            tempDict = pickle.load(f)
            cfg.MN_val_ids =  tempDict['ValIDs']
            cfg.MN_test_ids =  tempDict['TestIDs']
            cfg.MN_val_ids[0] = cfg.MN_val_ids[0][0:2]
            cfg.MN_test_ids[0] = cfg.MN_test_ids[0][0:2]'''

    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        if not hasattr(self, 'val_MN'):  # avoid unnecessary loading
            self.val_MN = get_data.MN_eval(self.cfg, self.cfg.MN_val_ids, 'val')
            self.test_MN = get_data.MN_eval(self.cfg, self.cfg.MN_test_ids, 'test')

    def val_dataloader(self):
        return DataLoader(self.val_MN, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_MN, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def show_sample(self, index=None):  # show a random sample for the data set
        if not index:  # get a random index if not specified
            index = np.random.randint(0, len(self.indices))
        data = get_data.NIH_DataSet(indices=self.indices, datatable=self.data_table)
        sample = data.__getitem__(index)
        img, label = sample['img'], sample['label']
        plt.imshow(img, 'gray')
        plt.title('Class: {}'.format(label))
        plt.show()


class Stroke(LightningDataModule):

    def __init__(self, cfg):
        super(Stroke, self).__init__()
        self.cfg = cfg
        # load data paths and indices
        # Stroke
        path = self.cfg.path.Stroke.IDs

        df_val = pd.read_csv(path['val'])
        df_test = pd.read_csv(path['test'])

        cfg.Stroke_val_ids = df_val["img_path"].apply(os.path.basename).to_list()
        cfg.Stroke_test_ids = df_test["img_path"].apply(os.path.basename).to_list()

        '''with open(path,'rb') as f :
            tempDict = pickle.load(f)
            cfg.val_ids =  tempDict['ValIDs']

            cfg.test_ids =  tempDict['TestIDs']
            cfg.val_ids = cfg.val_ids[0:1]
            print("stroke", cfg.val_ids)
            cfg.test_ids = cfg.test_ids[0:1]'''

    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        if not hasattr(self, 'val_set'):
            self.Stroke_val_set = get_data.Stroke(self.cfg, self.cfg.Stroke_val_ids, 'val')
            self.Stroke_test_set = get_data.Stroke(self.cfg, self.cfg.Stroke_test_ids, 'test')

    def val_dataloader(self):
        return DataLoader(self.Stroke_val_set, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True,
                          shuffle=False)

    def test_dataloader(self):
        test_DL = DataLoader(self.Stroke_test_set, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True,
                             shuffle=False)
        return test_DL


class Brats19(LightningDataModule):

    def __init__(self, cfg):
        super(Brats19, self).__init__()
        self.cfg = cfg
        # load data paths and indices
        # BraTS2019
        path = self.cfg.path.Brats19.IDs
        '''with open(path,'rb') as f :
            tempDict = pickle.load(f)
            cfg.Brats_val_ids =  tempDict['ValIDs']
            #print("cfg.Brats_val_ids",cfg.Brats_val_ids)
            cfg.Brats_test_ids =  tempDict['TestIDs']
            cfg.Brats_val_ids = cfg.Brats_val_ids[0:2]
            #print("cfg.Brats_val_ids[0]",cfg.Brats_val_ids[0])
            cfg.Brats_test_ids = cfg.Brats_test_ids[0:2]'''

        df_val = pd.read_csv(path['val'])
        df_test = pd.read_csv(path['test'])

        cfg.Brats19_val_ids = df_val["img_path"].apply(os.path.basename).to_list()
        cfg.Brats19_test_ids = df_test["img_path"].apply(os.path.basename).to_list()

    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        if not hasattr(self, 'val_set'):
            self.val_Brats19 = get_data.Brats19(self.cfg, self.cfg.Brats19_val_ids, 'val')
            self.test_Brats19 = get_data.Brats19(self.cfg, self.cfg.Brats19_test_ids, 'test')

    def val_dataloader(self):
        return DataLoader(self.val_Brats19, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True,
                          shuffle=False)

    def test_dataloader(self):
        Brats_test_DL = DataLoader(self.test_Brats19, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True,
                                   shuffle=False)
        return Brats_test_DL


class Brats20(LightningDataModule):

    def __init__(self, cfg):
        super(Brats20, self).__init__()
        self.cfg = cfg
        # load data paths and indices
        # BraTS2019
        path = self.cfg.path.Brats20.IDs
        '''with open(path,'rb') as f :
            tempDict = pickle.load(f)
            cfg.Brats_val_ids =  tempDict['ValIDs']
            #print("cfg.Brats_val_ids",cfg.Brats_val_ids)
            cfg.Brats_test_ids =  tempDict['TestIDs']
            cfg.Brats_val_ids = cfg.Brats_val_ids[0:2]
            #print("cfg.Brats_val_ids[0]",cfg.Brats_val_ids[0])
            cfg.Brats_test_ids = cfg.Brats_test_ids[0:2]'''

        df_val = pd.read_csv(path['val'])
        df_test = pd.read_csv(path['test'])

        cfg.Brats20_val_ids = df_val["img_path"].apply(os.path.basename)[0:100].to_list()
        cfg.Brats20_test_ids = df_test["img_path"].apply(os.path.basename)[0:100].to_list()

    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        if not hasattr(self, 'val_set'):
            self.val_Brats20 = get_data.Brats20(self.cfg, self.cfg.Brats20_val_ids, 'val')
            self.test_Brats20 = get_data.Brats20(self.cfg, self.cfg.Brats20_test_ids, 'test')

    def val_dataloader(self):
        return DataLoader(self.val_Brats20, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True,
                          shuffle=False)

    def test_dataloader(self):
        Brats_test_DL = DataLoader(self.test_Brats20, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True,
                                   shuffle=False)
        return Brats_test_DL


class MSLUB(LightningDataModule):

    def __init__(self, cfg):
        super(MSLUB, self).__init__()
        self.cfg = cfg
        # load data paths and indices
        # Stroke
        path = self.cfg.path.MSLUB.IDs

        df_val = pd.read_csv(path['val'])
        df_test = pd.read_csv(path['test'])

        cfg.MSLUB_val_ids = df_val["img_path"].apply(os.path.basename).to_list()
        cfg.MSLUB_test_ids = df_test["img_path"].apply(os.path.basename).to_list()

        '''with open(path,'rb') as f :
            tempDict = pickle.load(f)
            cfg.val_ids =  tempDict['ValIDs']

            cfg.test_ids =  tempDict['TestIDs']
            cfg.val_ids = cfg.val_ids[0:1]
            print("stroke", cfg.val_ids)
            cfg.test_ids = cfg.test_ids[0:1]'''

    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        if not hasattr(self, 'val_set'):
            self.MSLUB_val_set = get_data.MSLUB(self.cfg, self.cfg.MSLUB_val_ids, 'val')
            self.MSLUB_test_set = get_data.MSLUB(self.cfg, self.cfg.MSLUB_test_ids, 'test')

    def val_dataloader(self):
        return DataLoader(self.MSLUB_val_set, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True,
                          shuffle=False)

    def test_dataloader(self):
        test_DL = DataLoader(self.MSLUB_test_set, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True,
                             shuffle=False)
        return test_DL

