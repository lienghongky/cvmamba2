import logging
import torch
import torch.nn as nn
from os import path as osp
import sys
# for some possible IMPORT ERROR
# sys.path.append('/data1/guohang/MambaIR-main')
from modelkit.data import build_dataloader, build_dataset
from modelkit.model import build_model
from modelkit.utils import get_root_logger, get_time_str, make_exp_dirs
from modelkit.utils.options import dict2str, parse_options
from models import CVMambaIR


def test_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)
    torch.save(model.net_g,'CVMambaIRBase.pth')
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])


if __name__ == '__main__':

    # height = 256
    # width = 256
    # model = CVMambaIR(
    #     inp_channels=3,
    #     out_channels=3,
    #     dim=48,
    #     num_blocks=[4, 6, 6, 8],
    #     num_refinement_blocks=4,
    #     mlp_ratio=2.,
    #     bias=False,
    #     dual_pixel_task=False
    # )
    # # print(model)
    # x = torch.randn((1, 3, height, width))
    # print(x.shape)
    # y = model(x)
    # print(y.shape)

    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
