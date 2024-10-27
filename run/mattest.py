import os

import numpy as np
import torch.distributed as dist
from tqdm import tqdm
from skimage import img_as_ubyte
import scipy.io as sio

import torch
import torch.nn as nn

from PIL import Image

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

def setup(rank, world_size):
    # Initialize the process group
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()


def denoise_image(model, image_path):

    # load gt patches and init psnr ssim recorder
    filepath = 'ValidationGtBlocksSrgb.mat'
    img = sio.loadmat(filepath)

    # print(img['ValidationGtBlocksSrgb'])

    gt = np.float32(np.array(img['ValidationGtBlocksSrgb']))
    gt /= 255.
    # print('gt', gt.shape, gt.max(), gt.min())
    res = {'psnr': [], 'ssim': []}

    filepath = 'ValidationNoisyBlocksSrgb.mat'
  
    img = sio.loadmat(filepath)
    Inoisy = np.float32(np.array(img['ValidationNoisyBlocksSrgb']))

    Inoisy /= 255.
    restored = np.zeros_like(Inoisy)
    with torch.no_grad():
        for i in tqdm(range(40)):
            for k in range(32):
                noisy_patch = torch.from_numpy(Inoisy[i, k, :, :, :]).unsqueeze(0).permute(0, 3, 1, 2).cuda()
                restored_patch = model(noisy_patch)
                restored_patch = torch.clamp(restored_patch, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0)
                restored[i, k, :, :, :] = restored_patch
                # save psrn and ssim
                # print(type(restored_patch))  # torch.Tensor
                print('restored_patch', noisy_patch.shape)
                print('gt', gt[i, k].shape)

                
                psnr = compare_psnr(gt[i, k], restored_patch.numpy())
                ssim = compare_ssim(gt[i, k], restored_patch.numpy(), multichannel=True, channel_axis=2, data_range=1)
                print(f'{i} psnr %.2f ssim %.3f' % (psnr, ssim))
                res['psnr'].append(psnr)
                res['ssim'].append(ssim)
                if True:
                    gt_image = Image.fromarray((gt[i, k]*255).astype(np.uint8))  
                    gt_image.save(f'datasets/SIDD_Val/target_crops/patch_{i}_{k}_{psnr}.png')
                    noisy_image = Image.fromarray(noisy_patch.squeeze(0).permute(1, 2, 0).mul(255).byte().cpu().numpy())
                    noisy_image.save(f'datasets/SIDD_Val/input_crops/patch_{i}_{k}_{psnr}.png')
                

    print(f'mean {1} psnr %.2f ssim %.3f' % (np.mean(res['psnr']), np.mean(res['ssim'])))
    print(f'mean {1} psnr %.2f ssim %.3f' % (np.average(res['psnr']), np.average(res['ssim'])))

def main(rank, world_size, image_path):
    setup(rank, world_size)
    
    # Load the model
    model = torch.load('models/CVMambaIRBase.pth')
    
    # Check if model was wrapped in DistributedDataParallel
    if isinstance(model, nn.parallel.DistributedDataParallel):
        model = model.module
    

    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    model.eval()

    denoise_image(model, image_path)

    cleanup()
if __name__ == "__main__":
                 #SIDD/target_crops/0200_010_GP_01600_03200_5500_N_SRGB_010.PNG
    image_path = 'SIDD/input_crops/0200_010_GP_01600_03200_5500_N_SRGB_010.PNG'
    # image_path = 'image.mat'


    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main,
                                args=(world_size, image_path),
                                nprocs=world_size,
                                join=True)