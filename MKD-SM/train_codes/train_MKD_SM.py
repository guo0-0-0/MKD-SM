import argparse
import math
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from utils.utils import *
from data_loader.MKD_dataloader import load_dataloader
from train_models.model_MKD_SM import MultiSMNet
import os
from torch.nn.parallel import DistributedDataParallel

def save_network(net, save_path):

	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	if isinstance(net, nn.DataParallel) or isinstance(net, DistributedDataParallel):
		net = net.module
	state_dict = net.state_dict()
	for key, param in state_dict.items():
		if key.startswith('module.'):
			key = key[7:]   # remove unnecessary 'module.'
		state_dict[key] = param.cpu()
	torch.save(state_dict, save_path)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--seed', type=int, default=2022)
	parser.add_argument('--batch_size', type=int, default=256)
	parser.add_argument('--max_device_batch_size', type=int, default=256)
	parser.add_argument('--weight_decay', type=float, default=0.05)
	parser.add_argument('--warmup_epoch', type=int, default=10)
	parser.add_argument('--mask_ratio', type=float, default=0.0)
	parser.add_argument('--pretrained_model_path', type=int, default=0)

	args = parser.parse_args()
	setup_seed(args.seed)
	batch_size = args.batch_size
	load_batch_size = min(args.max_device_batch_size, batch_size)

	assert batch_size % load_batch_size == 0
	steps_per_update = batch_size // load_batch_size

	root_path = r'/media/MKD-SM/'

	train_experimentIDs = [7]
	test_experimentIDs = [10]

	in_chans = 2
	num_frames = 3

	down = 4
	lr = 1e-4

	snr = 5

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

	data_path = root_path + 'processed_data/'

	##### load Multi frames SM data and model
	train_loader, test_loader, _, _, _ = \
		load_dataloader(data_path, train_experimentIDs, test_experimentIDs, snr=snr,
								frames=num_frames, down=down, p=p, batch_size=load_batch_size)

	model = MultiSMNet(in_chans=in_chans, num_frames=num_frames,
						scale=down, feat_embed=feat_embed,
						img_size=img_size, patch_size=patch_size,
						embed_dim=embed_dim, num_head=en_head, num_layer=en_layer,
						out_channel=out_channel, kernel_size=kernel_size,
						n_conv_layer=n_conv, HR_channel=in_chans
						)

	model = model.to(device)
	model = torch.nn.DataParallel(model)
	optim = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
	writer = SummaryWriter(os.path.join('logs', 'reco_down' + str(down),
										'train' + str(train_experimentIDs[0]) +
										'_test' + str(test_experimentIDs[0]) +
										'_epoch' + str(total_epochs) +
										'_head' + str(en_head) +
										'_ly' + str(en_layer) +
										'_embed' + str(embed_dim) +
										'_k' + str(kernel_size) +
										'_' + 'L1_1e4' +
										'_snr' + str(snr) +
										'_frames' + str(num_frames) + pos_index))


	loss_fn = torch.nn.L1Loss(reduction='sum').to(device)


	step_count = 0
	optim.zero_grad()
	train_loss_list = []
	test_loss_list = []

	for epoch in range(total_epochs):
		model.train()
		losses = []
		train_step = len(train_loader)
		with tqdm(total=train_step, desc=f'Train Epoch {epoch+1}/{total_epochs}', postfix=dict, mininterval=0.3) as pbar:
			for LR_img, HR_img in iter(train_loader):
				step_count += 1
				LR_img, HR_img = LR_img.to(device), HR_img.to(device)
				pre_HR = model(LR_img)

				spa_loss = loss_fn(pre_HR, HR_img)
				spa_loss = spa_loss / np.prod(pre_HR.shape[1:])
				loss = spa_loss / LR_img.shape[0]

				loss.backward()
				optim.step()
				optim.zero_grad()
				losses.append(loss.item())

				pbar.set_postfix(**{'Train Loss' : np.mean(losses)})
				pbar.update(1)

		avg_loss = sum(losses) / len(losses)
		train_loss_list.append(avg_loss)
		print('epoch: ', epoch+1, 'mean loss: ', train_loss_list[-1])
		writer.add_scalar('train_resolution_avgloss', avg_loss, global_step=epoch+1)
		
		model.eval()
		with torch.no_grad():
			test_losses = []
			test_step = len(test_loader)
			with tqdm(total=test_step, desc=f'Val Epoch {epoch+1}/{total_epochs}', postfix=dict, mininterval=0.3) as pbar2:
				for test_LR_image, test_HR_image, _, _ in iter(test_loader):
					test_LR_image, test_HR_image = test_LR_image.to(device), test_HR_image.to(device)
					test_pre_HR = model(test_LR_image)

					test_spa_loss = loss_fn(test_pre_HR, test_HR_image)
					test_spa_loss = test_spa_loss / np.prod(test_pre_HR.shape[1:])
					test_loss = test_spa_loss / test_LR_image.shape[0]

					test_losses.append(test_loss.item())
					pbar2.set_postfix(**{'Val Loss' : np.mean(test_losses)})
					pbar2.update(1)
			avg_test_loss = sum(test_losses) / len(test_losses)
			test_loss_list.append(avg_test_loss)
		print('epoch: ', epoch+1, 'mean test loss: ', test_loss_list[-1])
		writer.add_scalar('test_resolution_avgloss', avg_test_loss, global_step=epoch+1)

	model_path = root_path + 'result/model_down' + str(down) + '/MKD_SM.pth'
	save_network(model, model_path)

