import scipy.io
import torch
import torch.nn.functional as F
import numpy as np
import pickle
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from scipy.ndimage import zoom
from einops import rearrange


transform = torch.from_numpy

class FreqSystemMatrixDataset(Dataset):

	def __init__(self, HR_SM, transform=None, down=2, mode='train', model_type='SRCNN'):
		self.mode = mode

		self.transform = transform
		self.down = down
		self.HR_SM = HR_SM

		self.HR_SM = self.HR_SM.transpose(0, 2, 1, 3, 4)
		print('self.HR_SM shape', self.HR_SM.shape)
		self.HR_shp = (self.HR_SM.shape[3], self.HR_SM.shape[4])

		self.LR_SM = self.down_sampling()
		print('self.LR_SM shape', self.LR_SM.shape)

		self.HR_size = self.HR_SM.shape[3:]

		self.HR_SM = rearrange(self.HR_SM, 'n d c h w -> (n d) c h w')
		self.LR_SM = rearrange(self.LR_SM, 'n d c h w -> (n d) c h w')
		self.origin_HR_SM = self.HR_SM
		print('self.origin_HR_SM', self.origin_HR_SM.shape)

		LR_mean = np.mean(self.HR_SM, (1, 2, 3)).reshape(self.HR_SM.shape[0], 1, 1, 1)
		LR_std = np.std(self.HR_SM, (1, 2, 3)).reshape(self.HR_SM.shape[0], 1, 1, 1)

		self.LR_mean, self.LR_std = LR_mean, LR_std
		self.LR_SM = (self.LR_SM - LR_mean) / LR_std
		self.HR_SM = (self.HR_SM - LR_mean) / LR_std
		print('self.LR_mean shape', self.LR_mean.shape)

		if model_type in ['SRCNN', 'VDSR']:
			self.LR_SM = torch.from_numpy(self.LR_SM).float()
			self.LR_SM = F.interpolate(self.LR_SM, scale_factor=self.down, mode='bicubic', align_corners=False)
			self.LR_SM = self.LR_SM.numpy()
			print('LR SM (after bicubic) shape: ', self.LR_SM.shape)


	def __len__(self):
		self.length = self.HR_SM.shape[0]
		return self.length

	def __getitem__(self, idx):
		HR_img, LR_img = self.HR_SM[idx], self.LR_SM[idx]
		HR_img = self.transform(HR_img).float()
		LR_img = self.transform(LR_img).float()

		if self.mode in ['test']:
			LR_mean, LR_std = self.LR_mean[idx], self.LR_std[idx]
			return LR_img, HR_img, LR_mean, LR_std
		else:
			return LR_img, HR_img

	def get_img_size(self):
		return self.HR_size

	def get_LR_mean_and_std(self):
		return self.LR_mean, self.LR_std, self.origin_HR_SM

	def down_sampling(self):
		sampling_indices = []
		for axis in range(2):
			indices = np.arange(self.HR_shp[axis])
			if self.HR_shp[axis] % self.down != 0:
				integrat_length = self.HR_shp[axis] - self.HR_shp[axis] % self.down
				rest_mid = int((self.HR_shp[axis] - integrat_length) / 2 + 0.5) + integrat_length - 1
				grid_indices = indices[: integrat_length][0::self.down]
				rest_ind = indices[rest_mid]
				down_sampling_indices = np.concatenate([grid_indices, [rest_ind]])
			else:
				down_sampling_indices = indices[0::self.down]
			sampling_indices.append(down_sampling_indices)

		LR_SM = self.HR_SM[:, :, :, sampling_indices[0]] \
			[:, :, :, :, sampling_indices[1]]
		return LR_SM


def load_freq_dataset(root_path, experimentIDs, down=2, snr=5, transform=None, mode='train', idx=None,  model_type='SRCNN'):
	all_SMs = []
	for exp_id in experimentIDs:
		cur_exp_SM_path = os.path.join(root_path, 'data' + str(exp_id) + '_SNR' + str(snr)) + '.pkl'
		cur_SM = pickle.load(open(cur_exp_SM_path, 'rb'))
		all_SMs.append(cur_SM)
	all_SMs = np.concatenate(all_SMs, 0)

	if idx is not None:
		all_SMs = all_SMs[idx]
	dataset = FreqSystemMatrixDataset(all_SMs, transform, down, mode=mode, model_type=model_type)
	return dataset


def load_dataloader(root_path, train_experimentIDs, test_experimentIDs, batch_size=256, down=2, snr=5, model_type='SRCNN'):

	train_dataset = load_freq_dataset(root_path, train_experimentIDs, down, snr, transform=transform, model_type=model_type)

	test_dataset = load_freq_dataset(root_path, test_experimentIDs, down, snr, transform=transform, mode='test', model_type=model_type)
	LR_mean, LR_std, origin_HR_SM = test_dataset.get_LR_mean_and_std()

	# loaders
	train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

	return train_loader, test_loader, LR_mean, LR_std, origin_HR_SM


