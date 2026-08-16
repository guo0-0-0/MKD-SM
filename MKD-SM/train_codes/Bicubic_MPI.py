
# ######################################################
# ##########Bicubic recostuction for MPI SM ############
# ######################################################

import numpy as np
from einops import rearrange
import pickle
from pylab import *
import time
import torch
import torch.nn.functional as F


# downsampling
def down_sampling(HR, scile0, down, HR_size):
    sampling_indices = []
    for axis in range(2):
        indices = np.arange(HR_size[axis])
        down_sampling_indices = indices[scile0::down]
        sampling_indices.append(down_sampling_indices)

    LR_SM = HR[:, sampling_indices[0]] \
        [:, :, sampling_indices[1]]
    return LR_SM


height, width = 36, 36
image = np.random.rand(height, width, 2).astype(np.float32)

root_path = r'/media/MKD-SM/'

SM_path = root_path + 'preprocessed_data/data10_SNR5.pkl'
origin_HR_SM = pickle.load(open(SM_path, 'rb'))

origin_HR_SM = origin_HR_SM.transpose(0, 2, 1, 3, 4)   # shape (f, d, c, h, w)
print('origin_HR_SM shape', origin_HR_SM.shape)

f, d, c, h, w = origin_HR_SM.shape
HR_size = (origin_HR_SM.shape[3], origin_HR_SM.shape[4])
print('HR_size: ', HR_size)

tr_HR_SM = rearrange(origin_HR_SM, 'f d c h w -> (f d c) h w')
print('tr_HR_SM shape: ', tr_HR_SM.shape)

scale_factor = 2 
LR_SM = down_sampling(tr_HR_SM, 1, scale_factor, HR_size)

LR_mean = np.mean(LR_SM, (1, 2)).reshape(LR_SM.shape[0], 1, 1)
LR_std = np.std(LR_SM, (1, 2)).reshape(LR_SM.shape[0], 1, 1)
print('LR_mean shape', LR_mean.shape)

norm_LR_SM = (LR_SM - LR_mean) / LR_std
print('norm_LR_SM shape', norm_LR_SM.shape)

norm_Bi_HR_SM = np.zeros(tr_HR_SM.shape)
start_total_time = time.time()

lr_tensor = torch.from_numpy(norm_LR_SM).unsqueeze(1).float().to('cuda')  # 添加 channel=1
hr_tensor = F.interpolate(lr_tensor, scale_factor=scale_factor, mode='bicubic', align_corners=False)

hr_tensor = F.interpolate(lr_tensor, scale_factor=scale_factor, mode='bicubic', align_corners=False)
norm_Bi_HR_SM = hr_tensor.squeeze(1).cpu().numpy()


end_total_time = time.time()
total_time = end_total_time - start_total_time  
print('total_time: ', total_time)
avg_time_per_image = (total_time / norm_Bi_HR_SM.shape[0]) * 1000  # 单位：毫秒
print('avg_time_per_image: ', avg_time_per_image)

Bi_HR_SM = norm_Bi_HR_SM * LR_std + LR_mean
print('Bi_HR_SM shape: ', Bi_HR_SM.shape)

Bi_HR_SM = rearrange(Bi_HR_SM, '(f d c) h w -> f c d h w', f=f, d=d, c=c)
print('Bi_HR_SM shape: ', Bi_HR_SM.shape)

trans_origin_HR_SM = origin_HR_SM.transpose(0, 2, 1, 3, 4)
print('trans_origin_HR_SM shape', trans_origin_HR_SM.shape)

comp_origin_HR_SM = trans_origin_HR_SM[:, 0, :, 5:-3, 5:-3] + 1j * trans_origin_HR_SM[:, 1, :, 5:-3, 5:-3]
comp_reco_HR_SM = Bi_HR_SM[:, 0, :, 5:-3, 5:-3] + 1j * Bi_HR_SM[:, 1, :, 5:-3, 5:-3]
print('comp_reco_HR_SM shape', comp_reco_HR_SM.shape)

vec_origin_HR_SM = comp_origin_HR_SM.reshape(comp_origin_HR_SM.shape[0], 1, -1)
vec_reco_HR_SM = comp_reco_HR_SM.reshape(comp_reco_HR_SM.shape[0], 1, -1)
print('vec_reco_HR_SM shape', vec_reco_HR_SM.shape)

N = vec_reco_HR_SM.shape[-1]
rmse = np.linalg.norm(vec_reco_HR_SM - vec_origin_HR_SM, 'fro', (1, 2)) / np.sqrt(N)
val_nrmse = rmse / np.max(np.abs(vec_origin_HR_SM), axis=(1, 2))

test_nrmses = val_nrmse.mean()
print('test nrmses: ', test_nrmses)


pred_HR_SM_poth = root_path + 'result_data5/bicubic_down' + str(scale_factor) + '_32.pkl'
# pred_HR_SM_poth = root_path + 'SM_reco/bicubic_down' + str(scale_factor) + '_32.pkl'
pickle.dump(comp_reco_HR_SM, open(pred_HR_SM_poth, 'wb'))



