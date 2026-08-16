import scipy.io
import torch
import numpy as np
import pickle
from torch.utils.data import DataLoader, Dataset
import os
import random
from einops import rearrange, repeat

class SM_Dataset(Dataset):

    def __init__(self, root_path, experimentIDs, snr, frames, down=2, p=0, mode='train', LR_mean=None, LR_std=None):
        
        self.mode = mode
        assert self.mode in ['train', 'test']
        
        self.down = down
        assert down in [2, 4, 8]

        print('p', p)

        self.all_SMs = []
        self.all_grads = []
        for exp_id in experimentIDs:
            SM_path = os.path.join(root_path,  'data' + str(exp_id) + '_SNR' + str(snr)) + '.pkl'
            SM_data = pickle.load(open(SM_path, 'rb'))
            self.all_SMs.append(SM_data)

        all_HR_SM = []
        all_multi_LR_SM = []
        all_LR_mean = []
        all_LR_std = []
        all_origin_HR = []
        for i in range(len(self.all_SMs)): 
                    
            self.origin_HR_SM = self.all_SMs[i]    # shape (f, c, d, h, w)
            self.origin_HR_SM = self.origin_HR_SM.transpose(0, 2, 1, 3, 4)   # shape (f, d, c, h, w)  
            f, d, c, h, w = self.origin_HR_SM.shape
            self.HR_size = (self.origin_HR_SM.shape[3], self.origin_HR_SM.shape[4])
            # print('self.origin_HR_SM shape', self.origin_HR_SM.shape)
            
            self.tr_HR_SM = rearrange(self.origin_HR_SM, 'f d c h w -> (f d) c h w') 
            all_origin_HR.append(self.tr_HR_SM)
            
            self.origin_LR_SM = np.zeros((f, d, c, int(h/down), int(w/down)))    # shape (f, d, c, h/2, w/2)
            self.LR_size = (self.origin_LR_SM.shape[3], self.origin_LR_SM.shape[4])

            # downsampling obtained LR_SM
            if frames == 2:    
                for i in range(self.origin_HR_SM.shape[1]):
                    if i % frames == 0:
                        self.origin_LR_SM[:, i, :, :, :] = self.down_sampling(self.origin_HR_SM[:, i, :, :, :], p[0], p[1])
                    else:
                        self.origin_LR_SM[:, i, :, :, :] = self.down_sampling(self.origin_HR_SM[:, i, :, :, :], p[2], p[3])
            if frames == 3:
                for i in range(self.origin_HR_SM.shape[1]):
                    if i % frames == 0:
                        self.origin_LR_SM[:, i, :, :, :] = self.down_sampling(self.origin_HR_SM[:, i, :, :, :], p[0], p[1])
                    elif i % frames == 1:
                        self.origin_LR_SM[:, i, :, :, :] = self.down_sampling(self.origin_HR_SM[:, i, :, :, :], p[2], p[3])
                    else:
                        self.origin_LR_SM[:, i, :, :, :] = self.down_sampling(self.origin_HR_SM[:, i, :, :, :], p[4], p[5])
            
            print('self.origin_LR_SM shape', self.origin_LR_SM.shape)
            self.tr_LR_SM = rearrange(self.origin_LR_SM, 'f d c h w -> (f d) c h w')   # shape (f d) c h w
            
            print('self.tr_LR_SM shape', self.tr_LR_SM.shape)

            LR_mean = np.mean(self.tr_LR_SM, (1, 2, 3)).reshape(self.tr_LR_SM.shape[0], 1, 1, 1)
            LR_std = np.std(self.tr_LR_SM, (1, 2, 3)).reshape(self.tr_LR_SM.shape[0], 1, 1, 1)

            self.norm_LR_SM = (self.tr_LR_SM - LR_mean) / LR_std
            self.norm_HR_SM = (self.tr_HR_SM - LR_mean) / LR_std

            all_HR_SM.append(self.norm_HR_SM)
            all_LR_mean.append(LR_mean)
            all_LR_std.append(LR_std)

            # obtain the multi-frames norm LR_SM
            self.norm_LR_SM = rearrange(self.norm_LR_SM, '(f d) c h w -> f d c h w', f=f, d=d) 
            self.new_LR_SM = np.zeros_like(self.norm_LR_SM)[:, :, np.newaxis, :, :, :]  # shape (f, d, 1, c, h, w)  3055, 27, 1, 2, 20, 20
            self.multi_LR_SM = np.tile(self.new_LR_SM, (1, 1, frames, 1, 1, 1))  # shape (f, d, frames, c, h, w) 37, 2753, 2, 2, 20, 20
            if frames == 2:
                print('frames: ', frames)
                for i in range(self.norm_LR_SM.shape[1]):
                    if i < (self.norm_LR_SM.shape[1] - 1):
                        # shape (f, d, frames, c, h, w) 3055, 27, 2, 2, 40, 40
                        a = self.norm_LR_SM[:, i, :, :, :]
                        b = self.norm_LR_SM[:, i+1, :, :, :]
                        self.multi_LR_SM[:, i, :, :, :, :] = np.concatenate((a[:,  np.newaxis, :, :, :], b[:,  np.newaxis, :, :, :]), axis=1)
                    else:
                        a = self.norm_LR_SM[:, i, :, :, :]
                        b = self.norm_LR_SM[:, i-1, :, :, :]
                        self.multi_LR_SM[:, i, :, :, :, :] = np.concatenate((a[:,  np.newaxis, :, :, :], b[:,  np.newaxis, :, :, :]), axis=1)
            else:
                print('frames: ', frames)
                for i in range(self.norm_LR_SM.shape[1]):
                    if i < (self.norm_LR_SM.shape[1] - 1):
                        if i == 0:
                            a = self.norm_LR_SM[:, i, :, :, :]
                            b = self.norm_LR_SM[:, i+1, :, :, :]
                            c = self.norm_LR_SM[:, i+2, :, :, :]
                            self.multi_LR_SM[:, i, :, :, :, :] = np.concatenate((a[:,  np.newaxis, :, :, :], b[:,  np.newaxis, :, :, :], c[:,  np.newaxis, :, :, :]), axis=1)
                        else:
                            # shape (f, d, frames, c, h, w) 3055, 27, 2, 2, 40, 40
                            a = self.norm_LR_SM[:, i, :, :, :]
                            b = self.norm_LR_SM[:, i-1, :, :, :]
                            c = self.norm_LR_SM[:, i+1, :, :, :]
                            self.multi_LR_SM[:, i, :, :, :, :] = np.concatenate((a[:,  np.newaxis, :, :, :], b[:,  np.newaxis, :, :, :], c[:,  np.newaxis, :, :, :]), axis=1)
                    else:
                        a = self.norm_LR_SM[:, i, :, :, :]
                        b = self.norm_LR_SM[:, i-1, :, :, :]
                        c = self.norm_LR_SM[:, i-2, :, :, :]
                        self.multi_LR_SM[:, i, :, :, :, :] = np.concatenate((a[:,  np.newaxis, :, :, :], b[:,  np.newaxis, :, :, :], c[:,  np.newaxis, :, :, :]), axis=1)
            print('self.multi_LR_SM shape', self.multi_LR_SM.shape)    
            self.new_multi_LR_SM = rearrange(self.multi_LR_SM, 'f d t c h w -> (f d) t c h w', f=f, d=d)
            print('self.new_multi_LR_SM shape', self.new_multi_LR_SM.shape) 
            all_multi_LR_SM.append(self.new_multi_LR_SM)
            
        self.HR_SM = np.concatenate(all_HR_SM, 0)
        self.LR_SM = np.concatenate(all_multi_LR_SM, 0)
        self.LR_mean = np.concatenate(all_LR_mean, 0)
        self.LR_std = np.concatenate(all_LR_std, 0)
        self.all_origin_HR = np.concatenate(all_origin_HR, 0)
        print('self.all_origin_HR shape', self.all_origin_HR.shape)

    def __len__(self):
        self.length = self.HR_SM.shape[0]
        return self.length

    def __getitem__(self, idx):
        HR_img, LR_img = self.HR_SM[idx], self.LR_SM[idx]
        HR_img = torch.from_numpy(HR_img).float()
        LR_img = torch.from_numpy(LR_img).float()

        if self.mode in ['test']:
            LR_mean, LR_std = self.LR_mean[idx], self.LR_std[idx]
            return LR_img, HR_img, LR_mean, LR_std
        else:
            return LR_img, HR_img

    def get_img_size(self):
        return self.LR_size, self.HR_size

    def get_LR_mean_and_std(self):
        return self.LR_mean, self.LR_std, self.all_origin_HR

    def down_sampling(self, HR, scile0, scile1):
        sampling_indices = []
        for axis in range(2):
            if axis == 0:
                indices = np.arange(self.HR_size[axis])
                down_sampling_indices = indices[scile0::self.down]
            else:
                indices = np.arange(self.HR_size[axis])
                down_sampling_indices = indices[scile1::self.down]
                
            sampling_indices.append(down_sampling_indices)
                
        LR_SM = HR[:, :, sampling_indices[0]] \
            [:, :, :, sampling_indices[1]]
        return LR_SM


def load_dataloader(root_path, train_experimentIDs, test_experimentIDs, snr, frames, down=2, batch_size=16, p=0):

    train_dataset = SM_Dataset(root_path, train_experimentIDs, snr, frames, down, p, mode='train', LR_mean=None, LR_std=None)
    test_dataset = SM_Dataset(root_path, test_experimentIDs, snr, frames, down, p, mode='test', LR_mean=None, LR_std=None)

    LR_mean, LR_std, origin_HR_SM = test_dataset.get_LR_mean_and_std()
    
    # loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, LR_mean, LR_std, origin_HR_SM



