import numpy as np
import h5py
import pickle
import re
import os
from pylab import *
import matplotlib.pyplot as plt
from einops import rearrange, repeat

def get_SM_img(mdf_file, snr_thres=5):
	'''
	Get the SM image data (2D) from the mdf data.
	Real value and Imag value are considered as two channels.
	'''

	# load mdf data
	f = h5py.File(mdf_file, 'r')

	# get background mask
	isBG = f['/measurement/isBackgroundFrame'][:].view(bool)

	# remove background and get SM with shape (C, K, H * W * D), e.g., (3, 3294, 33*33*27)
	SM = f['/measurement/data'][:, :, :, :].squeeze()[:, :, isBG == False]

	SM_size = f['/calibration/size']
	print('SM_size', SM_size)

	# get low SNR signals mask
	snr = f['calibration']['snr'][:, :, :].squeeze()
	print('snr shape', snr.shape)

	################################################################
	# 获取 SNR > snr_thres的数据
	mask = snr > snr_thres
	print('mask shape', mask.shape)

	ture_indices = np.argwhere(mask)
	print('ture_indices shape: ', ture_indices.shape)

	high_snr = snr[mask]
	high_snr = high_snr.reshape(-1, 1)
	print('high_snr: ', high_snr)
	print('high_snr shape', high_snr.shape)

	# shape(N, H*W*D)
	high_snr_SM = SM[mask]
	print('high_snr_SM shape', high_snr_SM.shape)

	# two channels respectively for Real value and Imag value
	Re_SM, Im_SM = high_snr_SM.real[:, np.newaxis, :], high_snr_SM.imag[:, np.newaxis, :]

	# shape(N, 2, H*W*D)
	SM_img = np.concatenate([Re_SM, Im_SM], 1)
	print('SM_img shape', SM_img.shape)

	# shape(N, 2, D, H, W )
	SM_img_input = rearrange(SM_img, 'n c (d h w) -> n c d h w ', d=SM_size[2], h=SM_size[0], w=SM_size[1])
	return SM_img_input, ture_indices, high_snr

if __name__ == '__main__':
	mdf_files = [r'/media/OpenMPI_SM/OpenData_7.mdf']
	snr_thres = 5

	for mdf_file in mdf_files:
		print(mdf_file)
		SM_img, SNR_indices, high_snr = get_SM_img(mdf_file, snr_thres)
		SM_img = np.pad(SM_img, ((0, 0), (0, 0), (0, 0), (2, 1), (2, 1)), 'constant', constant_values=(0, 0))

		experiment_idx = re.findall("\d+", mdf_file)[1]  
		experiment_SM_file = r'/media/preprocessed_data/' + 'data_' + experiment_idx + '.pkl'
		pickle.dump(SM_img, open(experiment_SM_file, 'wb'))
		print(SM_img.shape)
