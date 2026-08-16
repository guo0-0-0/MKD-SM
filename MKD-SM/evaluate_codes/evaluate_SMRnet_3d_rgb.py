import torch
from utils.utils import *
from data_loader.SMRnet_rgb_dataloader import load_dataloader
from train_models.other_Models import RRDBNet
import os
from einops import rearrange
import pickle
from matplotlib.colors import rgb_to_hsv
import time


def rgb_image_to_complex(rgb_image, Max_fu):
    # Remove the Alpha channel if it exists
    if rgb_image.shape[3] == 4:
        rgb_image = rgb_image[:, :, :, :3]

    # Convert RGB to HSV
    hsv_image = rgb_to_hsv(rgb_image)

    # Extract H, S, V channels
    hue = hsv_image[:, :, :, 0] * 2 * np.pi
    # hue = hue % (2 * np.pi)
    saturation = hsv_image[:, :, :, 1]
    value = hsv_image[:, :, :, 2] * Max_fu
    # value = value / np.max(value)
    # Calculate complex representation
    complex_data = value * np.exp(1j * hue)
    return complex_data


if __name__ == '__main__':

    setup_seed(2024)
    root_path = r'/media/MKD-SM/'

    train_experimentIDs = [7]
    test_experimentIDs = [10]

    down = 8

    snr = 5
    z_size = 27
    load_batch_size = 64

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_type = '3dSMRnet_rgb'
    data_path = root_path + 'processed_data/'

    # # load SM rgb data
    train_loader, test_loader, test_all_SMs_complex, test_amp = \
        load_dataloader(data_path, train_experimentIDs, test_experimentIDs, snr=snr,
                        down=down, batch_size=load_batch_size)

    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=9, gc=32, upscale=down, mode='CNA', act_type='leakyrelu',
                    upsample_mode='upconv', dim=3)
    
    model_path = root_path + 'result/down' + str(down) + '/3dSMRnet_rgb.pth'

    print('load resolution model success from: ', model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    pred_HR_SM = []
    with torch.no_grad():
        model.eval()
        start_total_time = time.time()
        for step, (LR_img, HR_img) in enumerate(test_loader):
            LR_img = LR_img.to(device)
            pred_HR_img = model(LR_img).cpu().numpy()
            pred_HR_SM.append(pred_HR_img)

        end_total_time = time.time()
        total_time = end_total_time - start_total_time 
        print('total_time: ', total_time)

        pred_HR_SM = np.concatenate(pred_HR_SM, 0)
        print('pred_HR_SM shape: ', pred_HR_SM.shape)

        avg_time_per_image = (total_time / pred_HR_SM.shape[0]) * 1000 
        print('avg_time_per_image: ', avg_time_per_image)

        pred_HR_SM = rearrange(pred_HR_SM, 'n c d h w -> n d h w c')

        RGB_complex_HR = []
        for i in range(pred_HR_SM.shape[0]):
            print(f"{i}-th")
            rgb_image = pred_HR_SM[i, :, :, :, :]
            # Convert RGB to complex5
            reconstructed_complex_data = rgb_image_to_complex(rgb_image, test_amp[i])
            RGB_complex_HR.append(reconstructed_complex_data)

        pre_HR_complex = np.stack(RGB_complex_HR, axis=0)   # complex data  [n d h w]
        print('pre_HR_complex shape', pre_HR_complex.shape)

        new_pre_HR_complex = pre_HR_complex[:, :, 5:-3, 5:-3]
        new_origin_HR_complex = test_all_SMs_complex[:, :, 5:-3, 5:-3]
        print('new_pre_HR_complex shape', new_pre_HR_complex.shape)

        # pre_HR_SM_path = root_path + 'SM_reco/SMRnet_3d_down' + str(down) + '.pkl'
        # pickle.dump(new_pre_HR_complex, open(pre_HR_SM_path, 'wb'))

        vec_pre_HR_complex = new_pre_HR_complex.reshape(new_pre_HR_complex.shape[0], 1, -1)
        vec_origin_HR_complex = new_origin_HR_complex.reshape(new_origin_HR_complex.shape[0], 1, -1)
        print('vec_pre_HR_complex shape', vec_pre_HR_complex.shape)

        N = vec_pre_HR_complex.shape[-1]
        rmse = np.linalg.norm(vec_pre_HR_complex - vec_origin_HR_complex, 'fro', (1, 2)) / np.sqrt(N)
        val_nrmse = rmse / np.max(np.abs(vec_origin_HR_complex), axis=(1, 2))

        nRMSE = val_nrmse.mean()
        print('nRMSE: ', nRMSE)

