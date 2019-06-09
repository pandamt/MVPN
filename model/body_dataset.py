# MVPN dataset: cope2(body)
import os, time
import numpy as np
import itertools as it
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model_settings import sub

# Device configuration
ROI_dir = '/gsfs0/data/fangmg/MVPD/data/ROIs/ses-movie/run_' 

# Implement the data loader.
class Body_Dataset(Dataset):
    """cope2(body) dataset."""
    def __init__(self, ROIs=[], GMs=[]):
        'Initialization'
        self.ROIs = []
        self.GMs = []

    def get_train(self, this_run=0, total_run=0):
        NULL = True # dataset is empty
        for run in it.chain(range(1,this_run), range(this_run+1,total_run+1)):
                EBA_data = np.load(ROI_dir + str(run) + '_ROIs/' + sub + '_cope2/EBA_80vox.npy')
                FBA_data = np.load(ROI_dir + str(run) + '_ROIs/' + sub + '_cope2/FBA_80vox.npy')
                STS_data = np.load(ROI_dir + str(run) + '_ROIs/' + sub + '_cope2/STS_80vox.npy')
                GM_data_run = np.load(ROI_dir + str(run) + '_ROIs/' + sub + '_cope2/GM_vox.npy')
                ROI_data_run = np.concatenate([EBA_data, FBA_data], 1)
                ROI_data_run = np.concatenate([ROI_data_run,STS_data], 1)
                if NULL:
                     ROI_data = ROI_data_run
                     GM_data = GM_data_run
                     NULL = False
                else: 
                     ROI_data = np.concatenate([ROI_data, ROI_data_run], 0) 
                     GM_data = np.concatenate([GM_data, GM_data_run], 0)
        
        # BatchNorm: dataset size modulo batch size is equal to 1        
        num_data = np.shape(ROI_data)[0]
        del_idx = np.random.randint(0, num_data)
        ROI_data = np.delete(ROI_data, del_idx, 0)
        GM_data = np.delete(GM_data, del_idx, 0)
 
        self.ROIs = torch.from_numpy(ROI_data)
        self.GMs = torch.from_numpy(GM_data)
        self.ROIs = self.ROIs.type(torch.FloatTensor)
        self.GMs = self.GMs.type(torch.FloatTensor)
      
    def get_test(self, this_run=0, total_run=0): 
        EBA_data = np.load(ROI_dir + str(this_run) + '_ROIs/' + sub + '_cope2/EBA_80vox.npy')
        FBA_data = np.load(ROI_dir + str(this_run) + '_ROIs/' + sub + '_cope2/FBA_80vox.npy')
        STS_data = np.load(ROI_dir + str(this_run) + '_ROIs/' + sub + '_cope2/STS_80vox.npy')
        ROI_data = np.concatenate([EBA_data, FBA_data], 1)
        ROI_data = np.concatenate([ROI_data, STS_data], 1)
        GM_data = np.load(ROI_dir + str(this_run) + '_ROIs/' + sub + '_cope2/GM_vox.npy') 
        'Convert ndarrays in sample to Tensors'
        self.ROIs = torch.from_numpy(ROI_data)
        self.GMs = torch.from_numpy(GM_data)
        self.ROIs = self.ROIs.type(torch.FloatTensor)
        self.GMs = self.GMs.type(torch.FloatTensor)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.ROIs)
    
    def __getitem__(self, idx):
        'Generates one sample of data'
        ROI = self.ROIs[idx]
        GM = self.GMs[idx]
        sample = {'ROI': ROI, 'GM': GM}
        return sample

