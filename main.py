import datetime
import os
import os.path as osp
import sys
import time
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from args import argument_parser, dataset_kwargs, optimizer_kwargs, lr_scheduler_kwargs
# from src import models # our folder with models

from utils.loggers import Logger
from utils.set_seed import set_random_seed
# global variables
parser = argument_parser()
args = parser.parse_args()


def main():
    global args

    set_random_seed(args.seed)
    if not args.use_avai_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False
    log_name = "log_test.txt" if args.evaluate else "log_train.txt"
    sys.stdout = Logger(osp.join(args.save_dir, log_name))
    print(f"==========\nArgs:{args}\n==========")

    if use_gpu:
        print(f"Currently using GPU {args.gpu_devices}")
        cudnn.benchmark = True
    else:
        warnings.warn("Currently using CPU, however, GPU is highly recommended")

#
# print("Initializing image data manager")
#     dm = ImageDataManager(use_gpu, **dataset_kwargs(args))
#     trainloader, testloader_dict = dm.return_dataloaders()


