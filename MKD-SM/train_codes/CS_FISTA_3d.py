# ######################################################
# ############## CS recostuction for MPI SM ############
# ######################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from scipy.linalg import lstsq
from sklearn.metrics import mean_squared_error
import torch
from tqdm import tqdm
import time
import pickle
from einops import rearrange


def build_sparse_phi(sampled_indices, n, device='cuda:1'):
	"""
	Phi, shape: [m, n]
	"""
	m = len(sampled_indices)
	row_idx = torch.arange(m, device=device, dtype=torch.long)
	col_idx = torch.tensor(sampled_indices, device=device, dtype=torch.long)
	values = torch.ones(m, device=device)
	indices = torch.stack([row_idx, col_idx])
	return torch.sparse_coo_tensor(indices, values, size=(m, n)).coalesce()


def fista_sparse_batch(Phi, y_batch, Psi, max_iter=50, tau=0.1, L=1.0, tol=1e-4):

    device = y_batch.device
    B, m = y_batch.shape
    n = Psi.shape[1]

    x = torch.zeros((B, n), device=device)
    z = x.clone()
    t = torch.ones(B, device=device)

    Psi_T = Psi.t()

    for _ in range(max_iter):
        x_old = x.clone()

        Psi_z = (Psi @ z.T).T
        Phi_Psi_z = torch.stack([torch.sparse.mm(Phi, Psi_z[i].unsqueeze(1)).squeeze() for i in range(B)])
        residual = Phi_Psi_z - y_batch
        grad = torch.stack([torch.sparse.mm(Phi.t(), residual[i].unsqueeze(1)).squeeze() for i in range(B)])

        z = z - (1.0 / L) * (grad @ Psi_T)
        x = torch.sign(z) * torch.clamp(torch.abs(z) - tau / L, min=0.0)

        t_new = (1 + torch.sqrt(1 + 4 * t**2)) / 2
        z = x + ((t - 1) / t_new).unsqueeze(1) * (x - x_old)
        t = t_new

        if torch.mean(torch.norm(x - x_old, dim=1) / (torch.norm(x_old, dim=1) + 1e-8)) < tol:
            break

    return x


def cs_fista_batch_reconstruction(origin_HR_SM, sampled_indices, device='cuda:1', max_iter=100, batch_size=256):

	device = torch.device(device)
	n_rows, n = origin_HR_SM.shape
	m = len(sampled_indices)

	Phi = build_sparse_phi(sampled_indices, n, device=device)
	Psi = torch.tensor(dct(np.eye(n), norm='ortho'), dtype=torch.float32, device=device)

	pre_HR_SM = []

	print(f"[INFO] device: {device}")
	start_time = time.time()
	for start in tqdm(range(0, n_rows, batch_size), desc='FISTA-CS Reconstruction'):
		end = min(start + batch_size, n_rows)
		# Get current batch observations
		batch_y = torch.tensor(origin_HR_SM[start:end][:, sampled_indices], dtype=torch.float32, device=device)

		mean = batch_y.mean(dim=1, keepdim=True)
		std = batch_y.std(dim=1, keepdim=True) + 1e-8 
		norm_batch_y = (batch_y - mean) / std

		# === FISTA reocstruction ===
		coeffs_batch = fista_sparse_batch(Phi, norm_batch_y, Psi, max_iter=max_iter, tau=0.1)

		# === Reconstruct Original Data ===
		batch_recon = coeffs_batch @ Psi.T

		batch_recon = batch_recon * std + mean  

		pre_HR_SM.append(batch_recon.cpu())

	duration = time.time() - start_time
	print(f"\n[INFO] all tims: {duration:.2f} 秒")

	pre_HR_SM = torch.cat(pre_HR_SM, dim=0).numpy()
	print(f"[INFO] Reconstruction completed, shape: {pre_HR_SM.shape}")
	return pre_HR_SM



if __name__ == '__main__':

	down = 4
	root_path = r'/media/MKD-SM/'
	sampled_path = root_path + 'poisson_disk_sampling/poisson_disk_sampled_down' + str(down) + '_33_all_slice_1d.pkl'

	sampled_indices = pickle.load(open(sampled_path, 'rb'))
	print('sampled_point: ', sampled_indices)
	print('len sampled_point: ', len(sampled_indices))

	SM_path = root_path + 'preprocessed_data/data10_SNR5.pkl'
	origin_HR_SM = pickle.load(open(SM_path, 'rb'))

	origin_HR_SM = origin_HR_SM[:, :, :, 4:-3, 4:-3]
	print('origin_HR_SM shape: ', origin_HR_SM.shape)

	f, c, d, h, w = origin_HR_SM.shape
	HR_size = (origin_HR_SM.shape[2], origin_HR_SM.shape[3], origin_HR_SM.shape[4])

	origin_HR_SM = rearrange(origin_HR_SM, 'f c d h w -> (f c) (d h w)')
	print('origin_HR_SM shape: ', origin_HR_SM.shape)

	pre_HR_SM = cs_fista_batch_reconstruction(origin_HR_SM, sampled_indices, device='cuda:1', max_iter=40,
											  batch_size=64)

	pre_HR_SM = rearrange(pre_HR_SM, '(f c) (d h w) -> f c d h w', f=f, c=c, d=d, h=h, w=w)

	# pre_HR_SM_path = root_path + 'SM_reco/CS_FISTA_down' + str(down) + '_33_3d.pkl'
	# pickle.dump(pre_HR_SM, open(pre_HR_SM_path, 'wb'))

	trans_origin_HR_SM = rearrange(origin_HR_SM, '(f c) (d h w) -> f c d h w', f=f, c=c, d=d, h=h, w=w)
	comp_origin_HR_SM = trans_origin_HR_SM[:, 0, :, :, :] + 1j * trans_origin_HR_SM[:, 1, :, :, :]
	comp_reco_HR_SM = pre_HR_SM[:, 0, :, :, :] + 1j * pre_HR_SM[:, 1, :, :, :]
	print('comp_reco_HR_SM shape', comp_reco_HR_SM.shape)

	vec_origin_HR_SM = comp_origin_HR_SM.reshape(comp_origin_HR_SM.shape[0], 1, -1)
	vec_reco_HR_SM = comp_reco_HR_SM.reshape(comp_reco_HR_SM.shape[0], 1, -1)
	print('vec_reco_HR_SM shape', vec_reco_HR_SM.shape)

	N = vec_reco_HR_SM.shape[-1]
	rmse = np.linalg.norm(vec_reco_HR_SM - vec_origin_HR_SM, 'fro', (1, 2)) / np.sqrt(N)
	val_nrmse = rmse / np.max(np.abs(vec_origin_HR_SM), axis=(1, 2))

	test_nrmses = val_nrmse.mean()
	print('test nrmses: ', test_nrmses)




