import scipy.io
import torch
import numpy as np
import pickle
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import random
from einops import rearrange
from matplotlib.colors import hsv_to_rgb

transform = torch.from_numpy


class FreqSystemMatrixDataset(Dataset):

    def __init__(self, HR_SM, SM_amp, all_SMs_complex, transform=None, down=2, mode='train', useTwoChannelInput=True,
                 LR_mean=None, LR_std=None):

        self.mode = mode
        self.useTwoChannelInput = useTwoChannelInput

        self.transform = transform
        self.down = down
        self.HR_SM = HR_SM
        self.SM_amp = SM_amp
        self.all_SMs_complex = all_SMs_complex
        print('self.HR_SM shape', self.HR_SM.shape)

        self.HR_SM = rearrange(self.HR_SM, 'n d h w c-> n c d h w')
        self.HR_shp = (self.HR_SM.shape[3], self.HR_SM.shape[4])
        print('self.HR_SM shape', self.HR_SM.shape)

        self.LR_SM = self.down_sampling()
        print('self.LR_SM shape', self.LR_SM.shape)

        self.HR_size, self.LR_size = self.HR_SM.shape[3:], self.LR_SM.shape[3:]


    def __len__(self):
        self.length = self.HR_SM.shape[0]
        return self.length

    def __getitem__(self, idx):
        HR_img, LR_img = self.HR_SM[idx], self.LR_SM[idx]
        HR_img = self.transform(HR_img).float()
        LR_img = self.transform(LR_img).float()
        return LR_img, HR_img

    def get_img_size(self):
        return self.LR_size, self.HR_size

    def get_origin_data(self):
        return  self.all_SMs_complex, self.SM_amp


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


def complex_array_to_hsv(complex_array, saturation):
    phase = np.angle(complex_array)

    magnitude = np.abs(complex_array)

    # Normalize phase to the range [0, 2π]
    phase = (phase + 2 * np.pi) % (2 * np.pi)

    # Normalize amplitude to the range [0, 1]
    Max_fu = np.max(magnitude)
    magnitude_normalized = magnitude / Max_fu

    # build HSV image
    hsv_image = np.zeros((phase.shape[0], phase.shape[1], phase.shape[2], 3))

    # H、S、V
    hsv_image[:, :, :, 0] = phase / (2 * np.pi)  # H
    hsv_image[:, :, :, 1] = saturation  # S
    hsv_image[:, :, :, 2] = magnitude_normalized  # V
    return hsv_image, Max_fu


def load_freq_dataset(root_path, experimentIDs, down=2, snr=5, transform=None, mode='train', LR_mean=None,
                      LR_std=None):

    # load the data IDs of the list and concatenate at axis=0
    all_SMs = []
    for exp_id in experimentIDs:
        cur_exp_SM_path = os.path.join(root_path, 'data' + str(exp_id) + '_snr' + str(snr)) + '.pkl'
        cur_SM = pickle.load(open(cur_exp_SM_path, 'rb'))
        all_SMs.append(cur_SM)
    all_SMs = np.concatenate(all_SMs, 0)


    # complerx data -> HSV date -> RGB data
    all_SMs_complex = all_SMs[:, 0, :, :, :] + 1j * all_SMs[:, 1, :, :, :]
    rgb_data = []
    rgb_fu = []
    for i in range(all_SMs_complex.shape[0]):
        print(f"Converting the {i}-th harmonic data....")
        # for i in range(complex_data.shape[0]):
        complex_data1 = all_SMs_complex[i, :, :, :]

        hsv_image, Max_fu = complex_array_to_hsv(complex_data1, 1.0)
        rgb_image = hsv_to_rgb(hsv_image)

        rgb_data.append(rgb_image)
        rgb_fu.append(Max_fu)

    all_rgb_data = np.stack(rgb_data, axis=0)
    all_rgb_amp = np.array(rgb_fu).reshape(-1, 1)

    dataset = FreqSystemMatrixDataset(all_rgb_data, all_rgb_amp, all_SMs_complex, transform, down, mode=mode, LR_mean=LR_mean, LR_std=LR_std)
    return dataset


def load_dataloader(root_path, train_experimentIDs, test_experimentIDs, batch_size=256, down=2, snr=5):

    train_dataset = load_freq_dataset(root_path, train_experimentIDs, down, snr, transform=transform)

    test_dataset = load_freq_dataset(root_path, test_experimentIDs, down, snr, transform=transform, mode='test')
    test_all_SMs_complex, test_amp = test_dataset.get_origin_data()

    # loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, test_all_SMs_complex, test_amp






