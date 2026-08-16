import torch
from utils.utils import *
from data_loader.other_model_dataloader import load_dataloader
from train_models.other_Models import VDSR_2channels, SRCNN_2channels, RRDBNet
import os
import time
from einops import rearrange
import pickle

if __name__ == '__main__':

	setup_seed(2024)
	root_path = r'/media/MKD-SM/'

	train_experimentIDs = [7]
	test_experimentIDs = [10]

	down = 8
	snr = 5

	z_size = 27

	load_batch_size = 256
	os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	model_type = 'SRCNN'
	# model_type = 'VDSR'

	data_path = root_path + 'preprocessed_data/'
	## load SM data
	train_loader, test_loader, LR_mean, LR_std, origin_HR_SM = \
		load_dataloader(data_path, train_experimentIDs, test_experimentIDs, snr=snr,
					down=down, batch_size=load_batch_size, model_type=model_type)

	model = SRCNN_2channels()  # VDSR_2channels
	model_path = root_path + 'result/down' + str(down) + '/SRCNN.pth' 
	print('load resolution model success from: ', model_path)

	model.load_state_dict(torch.load(model_path, map_location=device))
	model = model.to(device)

	pred_HR_SM = []
	with torch.no_grad():
		model.eval()
		start_total_time = time.time()
		for step, (LR_img, HR_img, _, _) in enumerate(test_loader):
			LR_img = LR_img.to(device)
			pred_HR_img = model(LR_img).cpu().numpy()
			pred_HR_SM.append(pred_HR_img)

		end_total_time = time.time()
		total_time = end_total_time - start_total_time 
		print('total_time: ', total_time)

		pred_HR_SM = np.concatenate(pred_HR_SM, 0)
		pred_HR_SM = pred_HR_SM * LR_std + LR_mean

		avg_time_per_image = (total_time / pred_HR_SM.shape[0]) * 1000 
		print('avg_time_per_image: ', avg_time_per_image)

		new_pred_HR_SM = pred_HR_SM[:, :, 5:-3, 5:-3]
		new_test_origin_HR_SM = origin_HR_SM[:, :, 5:-3, 5:-3]

		new_pred_HR_SM = rearrange(new_pred_HR_SM, '(f z) c h w -> f c z h w', z=z_size)
		new_test_origin_HR_SM = rearrange(new_test_origin_HR_SM, '(f z) c h w -> f c z h w', z=z_size)

		# pre_HR_SM_path = root_path + 'SM_reco/SRCNN_down' + str(down) + '.pkl'
        # pickle.dump(new_pre_HR_complex, open(pre_HR_SM_path, 'wb'))

		comp_reco_HR_SM = new_pred_HR_SM[:, 0, :, :, :] + 1j * new_pred_HR_SM[:, 1, :, :, :]
		comp_origin_HR_SM = new_test_origin_HR_SM[:, 0, :, :, :] + 1j * new_test_origin_HR_SM[:, 1, :, :, :]
		print('comp_reco_HR_SM shape', comp_reco_HR_SM.shape)

		vec_origin_HR_SM = comp_origin_HR_SM.reshape(comp_origin_HR_SM.shape[0], 1, -1)
		vec_reco_HR_SM = comp_reco_HR_SM.reshape(comp_reco_HR_SM.shape[0], 1, -1)
		print('vec_reco_HR_SM shape', vec_reco_HR_SM.shape)

		N = vec_reco_HR_SM.shape[-1]
		rmse = np.linalg.norm(vec_reco_HR_SM - vec_origin_HR_SM, 'fro', (1, 2)) / np.sqrt(N)
		val_nrmse = rmse / np.max(np.abs(vec_origin_HR_SM), axis=(1, 2))

		nRMSE = val_nrmse.mean()
		print('nRMSE: ', nRMSE)
