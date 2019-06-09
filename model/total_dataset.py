# MVPN dataset: cope1(face), cope2(body), cope3(object), cope4(scene)
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
class Total_Dataset(Dataset):
    """total dataset."""
    def __init__(self, ROIs=[], GMs=[]):
        'Initialization'
        self.ROIs = []
        self.GMs = []

    def get_train(self, this_run=0, total_run=0):
        NULL = True # dataset is empty
        for run in it.chain(range(1,this_run), range(this_run+1,total_run+1)):
        #for run in range(this_run+1,total_run+1):
                FFA_data = np.load(ROI_dir + str(run) + '_ROIs/' + sub + '_cope1/FFA_80vox.npy')
                fSTS_data = np.load(ROI_dir + str(run) + '_ROIs/' + sub + '_cope1/STS_80vox.npy')
                OFA_data = np.load(ROI_dir + str(run) + '_ROIs/' + sub + '_cope1/OFA_80vox.npy')
                EBA_data = np.load(ROI_dir + str(run) + '_ROIs/' + sub + '_cope2/EBA_80vox.npy')
                FBA_data = np.load(ROI_dir + str(run) + '_ROIs/' + sub + '_cope2/FBA_80vox.npy')
                bSTS_data = np.load(ROI_dir + str(run) + '_ROIs/' + sub + '_cope2/STS_80vox.npy')
                MTG_data = np.load(ROI_dir + str(run) + '_ROIs/' + sub + '_cope3/MTG_80vox.npy')
                MFA_data = np.load(ROI_dir + str(run) + '_ROIs/' + sub + '_cope3/MFA_80vox.npy')
                PPA_data = np.load(ROI_dir + str(run) + '_ROIs/' + sub + '_cope4/PPA_80vox.npy')
                TOS_data = np.load(ROI_dir + str(run) + '_ROIs/' + sub + '_cope4/TOS_80vox.npy')
                RSP_data = np.load(ROI_dir + str(run) + '_ROIs/' + sub + '_cope4/RSP_80vox.npy')	
                GM_data_run = np.load(ROI_dir + str(run) + '_ROIs/' + sub + '_cope1/GM_vox.npy')
                ROI_data_run = np.concatenate([FFA_data, fSTS_data], 1)
                ROI_data_run = np.concatenate([ROI_data_run, OFA_data], 1)
                ROI_data_run = np.concatenate([ROI_data_run, EBA_data], 1)
                ROI_data_run = np.concatenate([ROI_data_run, FBA_data], 1)
                ROI_data_run = np.concatenate([ROI_data_run, bSTS_data], 1)  
                ROI_data_run = np.concatenate([ROI_data_run, MTG_data], 1)
                ROI_data_run = np.concatenate([ROI_data_run, MFA_data], 1) 
                ROI_data_run = np.concatenate([ROI_data_run, PPA_data], 1) 
                ROI_data_run = np.concatenate([ROI_data_run, TOS_data], 1)
                ROI_data_run = np.concatenate([ROI_data_run, RSP_data], 1)

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
        FFA_data = np.load(ROI_dir + str(this_run) + '_ROIs/' + sub + '_cope1/FFA_80vox.npy')
        fSTS_data = np.load(ROI_dir + str(this_run) + '_ROIs/' + sub + '_cope1/STS_80vox.npy')
        OFA_data = np.load(ROI_dir + str(this_run) + '_ROIs/' + sub + '_cope1/OFA_80vox.npy')
        EBA_data = np.load(ROI_dir + str(this_run) + '_ROIs/' + sub + '_cope2/EBA_80vox.npy')
        FBA_data = np.load(ROI_dir + str(this_run) + '_ROIs/' + sub + '_cope2/FBA_80vox.npy')
        bSTS_data = np.load(ROI_dir + str(this_run) + '_ROIs/' + sub + '_cope2/STS_80vox.npy')
        MTG_data = np.load(ROI_dir + str(this_run) + '_ROIs/' + sub + '_cope3/MTG_80vox.npy')
        MFA_data = np.load(ROI_dir + str(this_run) + '_ROIs/' + sub + '_cope3/MFA_80vox.npy')
        PPA_data = np.load(ROI_dir + str(this_run) + '_ROIs/' + sub + '_cope4/PPA_80vox.npy')
        TOS_data = np.load(ROI_dir + str(this_run) + '_ROIs/' + sub + '_cope4/TOS_80vox.npy')
        RSP_data = np.load(ROI_dir + str(this_run) + '_ROIs/' + sub + '_cope4/RSP_80vox.npy')	
        GM_data = np.load(ROI_dir + str(this_run) + '_ROIs/' + sub + '_cope1/GM_vox.npy') 
        ROI_data = np.concatenate([FFA_data, fSTS_data], 1)
        ROI_data = np.concatenate([ROI_data, OFA_data], 1)
        ROI_data = np.concatenate([ROI_data, EBA_data], 1)
        ROI_data = np.concatenate([ROI_data, FBA_data], 1)
        ROI_data = np.concatenate([ROI_data, bSTS_data], 1)
        ROI_data = np.concatenate([ROI_data, MTG_data], 1)
        ROI_data = np.concatenate([ROI_data, MFA_data], 1)
        ROI_data = np.concatenate([ROI_data, PPA_data], 1)
        ROI_data = np.concatenate([ROI_data, TOS_data], 1)
        ROI_data = np.concatenate([ROI_data, RSP_data], 1)  
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

