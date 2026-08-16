import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from utils.utils import *
from data_loader.SMRnet_rgb_dataloader import load_dataloader
from train_models.other_Models import RRDBNet
from torch.optim import lr_scheduler
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
	parser.add_argument('--batch_size', type=int, default=8)
	parser.add_argument('--weight_decay', type=float, default=0.05)
	parser.add_argument('--warmup_epoch', type=int, default=10)
	parser.add_argument('--max_device_batch_size', type=int, default=256)
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

	down = 4
	snr = 5
	total_epochs = 100

	os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	model_type = '3dSMRnet_rgb'   #  3dSMRnet_rgb 

	data_path = root_path + 'preprocessed_data/'

	# load SM  rgb data
	train_loader, test_loader, test_all_SMs_complex, test_amp = \
		load_dataloader(data_path, train_experimentIDs, test_experimentIDs, snr=snr,
						down=down, batch_size=load_batch_size)

	lr = 1e-5
	model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=9, gc=32, upscale=down, mode='CNA', act_type='leakyrelu', upsample_mode='upconv', dim=3)
	model = model.to(device)
	model = torch.nn.DataParallel(model)
	optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0, betas=(0.9, 0.999))
	scheduler = lr_scheduler.MultiStepLR(optim, milestones=[30, 60], gamma=0.5)


	loss_fn = torch.nn.L1Loss(reduction='sum').to(device)

	writer = SummaryWriter(os.path.join('logs', 'cross', 'reco_down' + str(down),
										'train' + str(train_experimentIDs[0]) +
										'_test' + str(test_experimentIDs[0]) +
										'_epoch' + str(total_epochs) +
										'_' + 'L1_1e5' +
										'_snr' + str(snr) + '_' + model_type))

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

		scheduler.step()
		avg_loss = sum(losses) / len(losses)
		train_loss_list.append(avg_loss)
		print('epoch: ', epoch+1, 'mean loss: ', train_loss_list[-1])
		writer.add_scalar('train_resolution_avgloss', avg_loss, global_step=epoch+1)

		model.eval()
		with torch.no_grad():
			test_losses = []
			test_step = len(test_loader)
			with tqdm(total=test_step, desc=f'Val Epoch {epoch+1}/{total_epochs}', postfix=dict, mininterval=0.3) as pbar2:
				for test_LR_image, test_HR_image in iter(test_loader):
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

	model_path = root_path + 'result/down' + str(down) + '/3dSMRnet_rgb.pth'
	save_network(model, model_path)
