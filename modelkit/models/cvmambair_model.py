import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
from modelkit.utils.registry import MODEL_REGISTRY


from modelkit.archs import build_network
from modelkit.models.base_model import BaseModel
from modelkit.utils import get_root_logger, imwrite, tensor2img

loss_module = importlib.import_module('modelkit.losses')
metric_module = importlib.import_module('modelkit.metrics')

import os
import random
import numpy as np
import cv2
import torch.nn.functional as F
from functools import partial
from modelkit.utils import Mixing_Augment


@MODEL_REGISTRY.register()
class CVMambaIRDenoising(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(CVMambaIRDenoising, self).__init__(opt)

        # define network
        if self.is_train:
            self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
            if self.mixing_flag:
                mixup_beta       = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
                use_identity     = self.opt['train']['mixing_augs'].get('use_identity', False)
                self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        self.net_g = build_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(
                self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            raise ValueError('pixel loss are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        preds = self.net_g(self.lq)
        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        loss_dict = OrderedDict()
        # pixel loss
        l_pix = 0.
        for pred in preds:
            l_pix += self.cri_pix(pred, self.gt)

        loss_dict['l_pix'] = l_pix

        l_pix.backward()
        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def process_input(self, img, model_input_size):
        
        """
        Process the input image to fit the model's input size by applying necessary patching and padding.
        
        Parameters:
        - img: Input image tensor of shape (N, C, H, W).
        - model_input_size: Tuple (H, W) representing the input size expected by the model.
        
        Returns:
        - List of padded patches ready for model inference.
        """
        # save input image to file
        # imwrite(tensor2img([img], rgb2bgr=True), 'input_img.png')
        # Extract model input height and width
        model_input_height, model_input_width = model_input_size
        model_input_height -= 2
        model_input_width -= 2
        
        # Get original image size
        _, _, h, w = img.size()
        
        # List to hold patches
        patches = []

        # Calculate padding needed to make the image size divisible by model input size
        pad_h = (model_input_height - (h % model_input_height)) % model_input_height
        pad_w = (model_input_width - (w % model_input_width)) % model_input_width

        # Pad the original image to make it divisible by model input size, considering the 1px padding
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
        
        # New height and width after padding
        _, _, padded_h, padded_w = img.size()
        
        # save paadded image to file 
        # imwrite(tensor2img([img], rgb2bgr=True), 'padded_img.png')

        # Iterate over the padded image in patches
        for i in range(0, padded_h, model_input_height):
            for j in range(0, padded_w, model_input_width):
                # Extract patch
                patch = img[:, :, i:i + model_input_height, j:j + model_input_width]
                
                # Calculate additional padding needed for each patch to fit the model input size
                patch_h, patch_w = patch.size(2), patch.size(3)
                
                patch = F.pad(patch, (1, 1, 1, 1), mode='reflect')
                
                if hasattr(self, 'net_g_ema'):
                    self.net_g_ema.eval()
                    with torch.no_grad():
                        pred = self.net_g_ema(patch)
                    if isinstance(pred, list):
                        pred = pred[-1]
                    
                else:
                    self.net_g.eval()
                    with torch.no_grad():
                        pred = self.net_g(patch)
                    if isinstance(pred, list):
                        pred = pred[-1]
                # remove the padding around output
                pred = pred[:, :, 1:-1, 1:-1]
                patches.append(pred)
                # save patch to file
                # imwrite(tensor2img([pred], rgb2bgr=True), f'patch_{i}_{j}.png')
        
        def unpatch_image(patches, original_height, original_width, patch_size):
            """
            Reconstructs the original image from a list of processed patches.
            
            Parameters:
            - patches: List of patches (each patch tensor has shape (C, patch_H, patch_W)) from the model output.
            - original_height: Original height of the full image before patching.
            - original_width: Original width of the full image before patching.
            - patch_size: Size of each patch (H, W) after padding was added.
            
            Returns:
            - Full image reconstructed from patches, resized to the original dimensions.
            """
            # Determine the number of patches along the height and width
            patches_per_row = original_width // patch_size[1]
            patches_per_col = original_height // patch_size[0]
            
            # Initialize an empty tensor for the reconstructed image
            _, C, patch_H, patch_W = patches[0].size()  # Get channel and patch size from first patch
            full_image = torch.zeros((C, patches_per_col * patch_H, patches_per_row * patch_W))
            
            # Fill in the full image tensor with patches
            patch_idx = 0
            for i in range(patches_per_col):
                for j in range(patches_per_row):
                    y_start = i * patch_H
                    y_end = y_start + patch_H
                    x_start = j * patch_W
                    x_end = x_start + patch_W
                    full_image[:, y_start:y_end, x_start:x_end] = patches[patch_idx]
                    patch_idx += 1
            
            # Crop to original size in case padding was added during patching
            full_image = full_image[:, :original_height, :original_width]
            
            return full_image
        img = unpatch_image(patches, padded_h, padded_w, (model_input_height-2, model_input_width-2))
        # save image to file
        # imwrite(tensor2img([img], rgb2bgr=True), 'output_img.png')
        self.output = img[:, 0:h, 0:w]

    def pad_test(self, window_size): 
        self.process_input(self.lq, (window_size, window_size))
        return       
        print(f"pad_test w {window_size} \nlq {self.lq.shape}")
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        print("\n\nonpad_test: ",img.shape )
        if img is None:
            img = self.lq  

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred = self.net_g(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
            self.net_g.train()

    


    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr=True, use_image=True):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        # pbar = tqdm(total=len(dataloader), unit='image')

        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                
                if self.opt['is_train']:
                    
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')
                    
                    save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}_gt.png')
                else:
                    
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png')
                    save_gt_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_gt.png')
                    
                imwrite(sr_img, save_img_path)
                imwrite(gt_img, save_gt_img_path)
                # Save Input image
                if 'lg' in visuals:
                    save_lg_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_lg.png')
                    lg_img = tensor2img([visuals['lg']], rgb2bgr=rgb2bgr)
                    imwrite(lg_img, save_lg_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
