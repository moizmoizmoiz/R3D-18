import datetime
import os
import os.path as osp
import sys
import time
import warnings
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn as nn
from args import argument_parser, dataset_kwargs, optimizer_kwargs, lr_scheduler_kwargs
# from src import models # our folder with models
from src.dataloader import VideoDataset
from src.loggers import Logger
from src.set_seed import set_random_seed

# global variables
parser = argument_parser()
args = parser.parse_args()


def main():
    global args

    set_random_seed(args.seed)  # we set the default as 2222
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

    print("Transforming Data")
    data_transforms = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
    ])

    dataset = VideoDataset(root_dir='/content/HMDB_simp', frames_per_clip=3,  # TODO: root_dir as variable
                           transform=data_transforms)  # frames=3 because r3d takes only 3 frames
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)  # TODO: Variable

    train_indexes, test_indexes = train_test_split(range(len(dataset)), test_size=0.2, shuffle=True)  # TODO variable
