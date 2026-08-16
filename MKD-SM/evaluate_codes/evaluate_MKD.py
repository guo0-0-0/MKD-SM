import torch
from utils.utils import *
from data_loader.MKD_dataloader import load_dataloader
from train_models.model_MKD_SM import MultiSMNet
from einops import rearrange
import os
from pylab import *
import time
import pickle


if __name__ == '__main__':

	setup_seed(2024)
	root_path = r'/media/MDK-SM/'

	train_experimentIDs = [7]
	test_experimentIDs = [10]

	in_chans = 2
	num_frames = 3

	down = 4
	snr = 5
	z_size = 27

	model_type = 'MKD'

	batch_size = 256
	feat_embed = 32
	embed_dim = 48
	en_head = 6
	en_layer = 4

	if down == 8:
		out_channel = 64
	else:
		out_channel = 32
	kernel_size = 3
	n_conv = 3

	if down == 2:
		if num_frames == 2:
			p = [0, 0, 1, 1]  # p1:(0, 0) p2:(1, 1)
			pos_index = '_p2_' + str(p[2]) + str(p[3])
		else:
			p = [0, 0, 1, 1, 0, 1]  # p1:(0, 0) p2:(1, 1) p3:(0, 1)
			pos_index = '_p2_' + str(p[2]) + str(p[3]) + '_p3_' + str(p[4]) + str(p[5])
		img_size = 20
		patch_size = 2
		total_epochs = 100
	elif down == 4:
		if num_frames == 2:
			p = [0, 0, 2, 2]  # p1:(0, 0) p2:(2, 2)
			pos_index = '_p2_' + str(p[2]) + str(p[3])
		else:
			p = [0, 0, 2, 2, 0, 3]  # p1:(0, 0) p2:(2, 2) p3:(0, 3)
			pos_index = '_p2_' + str(p[2]) + str(p[3]) + '_p3_' + str(p[4]) + str(p[5])
		img_size = 10
		patch_size = 1
		total_epochs = 100
	else:
		if num_frames == 2:
			p = [5, 2, 3, 5]  # p1:(5, 2) p2:(3, 5)
			pos_index = '_p1_' + str(p[0]) + str(p[1]) + '_p2_' + str(p[2]) + str(p[3])
		else:
			p = [0, 0, 5, 2, 3, 5]  # p1:(0, 0) p2:(5, 2) p3:(3, 5)
			pos_index = '_p2_' + str(p[2]) + str(p[3]) + '_p3_' + str(p[4]) + str(p[5])
		img_size = 5
		patch_size = 1
		total_epochs = 100

	os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	data_path = root_path + 'processed_data/three_filter(8910)/real_data_40'

	## load multi frames SM data
	train_loader, test_loader, LR_mean, LR_std, origin_HR_SM = \
		load_dataloader(data_path, train_experimentIDs, test_experimentIDs, snr=snr,
								frames=num_frames, down=down, p=p, batch_size=batch_size)

	model = MultiSMNet(in_chans=in_chans, num_frames=num_frames,
					   scale=down, feat_embed=feat_embed,
					   img_size=img_size, patch_size=patch_size,
					   embed_dim=embed_dim, num_head=en_head, num_layer=en_layer,
					   out_channel=out_channel, kernel_size=kernel_size,
					   n_conv_layer=n_conv, HR_channel=in_chans).to(device)

	model_path = root_path + 'result/model_down' + str(down) + '/MKD_SM.pth'
	print('load resolution model success from: ', model_path)

	model.load_state_dict(torch.load(model_path, map_location=device))

	os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model = model.to(device)

	pred_HR_SM, test_nrmses = [], []

	with torch.no_grad():
		model.eval()
		start_total_time = time.time()

		for step, (LR_img, HR_img, _, _) in enumerate(test_loader):
			LR_img = LR_img.to(device)
			pred_HR_img = model(LR_img) 
			pred_HR_SM.append(pred_HR_img.cpu().numpy())

		end_total_time = time.time()
		total_time = end_total_time - start_total_time  
		print('total_time: ', total_time)

		pred_HR_SM = np.concatenate(pred_HR_SM, 0)
		pred_HR_SM = pred_HR_SM * LR_std + LR_mean
		print('pred_HR_SM shape', pred_HR_SM.shape)

		avg_time_per_image = (total_time / pred_HR_SM.shape[0]) * 1000  #
		print('avg_time_per_image: ', avg_time_per_image)

		new_pred_HR_SM = np.zeros(
			(pred_HR_SM.shape[0], pred_HR_SM.shape[1], pred_HR_SM.shape[2]-8, pred_HR_SM.shape[3]-8))
		new_test_origin_HR_SM = np.zeros(
			(origin_HR_SM.shape[0], origin_HR_SM.shape[1], origin_HR_SM.shape[2]-8, origin_HR_SM.shape[3]-8))

		new_pred_HR_SM[:, :, :, :] = pred_HR_SM[:, :, 5:-3, 5:-3]
		new_test_origin_HR_SM[:, :, :, :] = origin_HR_SM[:, :, 5:-3, 5:-3]

		new_pred_HR_SM = rearrange(new_pred_HR_SM, '(f z) c h w -> f c z h w', z=z_size)
		new_test_origin_HR_SM = rearrange(new_test_origin_HR_SM, '(f z) c h w -> f c z h w', z=z_size)


		# pre_HR_SM_path = root_path + 'SM_reco/MKD_down' + str(down) + '.pkl'
		# pickle.dump(new_pred_HR_SM, open(pre_HR_SM_path, 'wb'))

		comp_reco_HR_SM = new_pred_HR_SM[:, 0, :, :, :] + 1j * new_pred_HR_SM[:, 1, :, :, :]
		comp_origin_HR_SM = new_test_origin_HR_SM[:, 0, :, :, :] + 1j * new_test_origin_HR_SM[:, 1, :, :, :]
		print('comp_reco_HR_SM shape', comp_reco_HR_SM.shape)

		vec_origin_HR_SM = comp_origin_HR_SM.reshape(comp_origin_HR_SM.shape[0], 1, -1)
		vec_reco_HR_SM = comp_reco_HR_SM.reshape(comp_reco_HR_SM.shape[0], 1, -1)
		print('vec_reco_HR_SM shape', vec_reco_HR_SM.shape)

		N = vec_reco_HR_SM.shape[-1]
		rmse = np.linalg.norm(vec_reco_HR_SM - vec_origin_HR_SM, 'fro', (1, 2)) / np.sqrt(N)
		val_nrmse = rmse / np.max(np.abs(vec_origin_HR_SM), axis=(1, 2))

		print('nRMSE : ', val_nrmse.mean())
	
