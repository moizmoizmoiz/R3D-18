import datetime
import os.path as osp
import sys
import time
from tqdm.notebook import tqdm
from torchvision.models.video import R3D_18_Weights
from torch import optim
from src.tools import count_num_param
from sklearn.model_selection import train_test_split
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from args import argument_parser
from src.dataloader import VideoDataset
from src.loggers import Logger
from src.set_seed import set_random_seed
from src.train import train
from src.test import test

# global variables
parser = argument_parser()
args = parser.parse_args()
logger = Logger()
writer = logger.create_summary_writer()

def main():
    global args
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d--%H%M%S")
    # log_dir = "/content/drive/MyDrive/TensorBoard_Logs/" + formatted_datetime +"_" +args.name
    # writer = SummaryWriter(log_dir=log_dir)

    set_random_seed(args.seed)  # we set the default as 2222
    # if not args.use_avai_gpus:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    # use_gpu = torch.cuda.is_available()
    # if args.use_cpu:
    #     use_gpu = False
    log_name = formatted_datetime+"_"+args.name+".txt"
    sys.stdout = Logger(osp.join(args.save_dir, log_name))
    print(f"==========\nArgs:{args}\n==========")

    # if use_gpu:
    #     print(f"Currently using GPU {args.gpu_devices}")
    #     cudnn.benchmark = True
    # else:
    #     warnings.warn("Currently using CPU, however, GPU is highly recommended")

    print("Defined Transformations")
    data_transforms = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Add normalization here
    ])
    print("==========")
    dataset = VideoDataset(root_dir=args.root, frames_per_clip=3,
                           transform=data_transforms)  # frames=3 because r3d takes only 3 frames
    print("==========")
    train_indexes, test_indexes = train_test_split(range(len(dataset)), test_size=args.split, shuffle=True)

    train_set = torch.utils.data.Subset(dataset, train_indexes)
    test_set = torch.utils.data.Subset(dataset, test_indexes)
    print("Defining Model")
    model = torchvision.models.video.r3d_18(weights=R3D_18_Weights.DEFAULT)

    del model.fc

    model.fc = nn.Sequential(
    nn.Linear(512, 25)
    )
    nn.init.normal_(model.fc[0].weight, mean=0.0, std=0.002)

    model.classi = nn.Sequential(
    nn.LogSoftmax(dim=1)
    )

    final_layer_params = list(model.fc.parameters())
    other_params = [param for name, param in model.named_parameters() if not name.startswith('fc')]


    print("Model size: {:.3f} M".format(count_num_param(model)))
    # writer.add_graph(model, use_strict_trace=False)
    if args.mprint: print(model)
    print("==========")
    print("Loading Dataloaders...")
    loader_args = dict(batch_size=args.batch, num_workers=args.workers, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)
    print("==========")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device: ", device)
    model = model.to(device)
    print("==========")
    print("Optimizer Defined...")
    # Create separate parameter groups for the final layer and other layers
    optimizer = optim.Adam([{'params': other_params, 'lr': args.lr}, {'params': final_layer_params, 'lr': args.lr_fc}], weight_decay=args.decay)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    print("==========")
    time_start = time.time()
    for epoch in tqdm(range(args.epochs)):
        loss_avg = train(model,
                         train_loader,
                         optimizer,
                         device)
        print("loss: {:.4f} for epoch: {}/{}".format(loss_avg, epoch + 1, args.epochs))
        writer.add_scalar('Loss/Train', loss_avg, epoch)

    test(model,
         test_loader,
         device)
    writer.add_text('LR', str(args.lr))
    writer.add_text('Final Layer LR', str(args.lr_fc))
    writer.add_text('Weight Decay', str(args.decay))
    writer.add_text('Epochs', str(args.epochs))
    writer.add_text('Batch Size', str(args.batch))
    writer.add_text('Split', str(args.split))

    elapsed = round(time.time() - time_start)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print(f"Elapsed {elapsed}")

    writer.close()


if __name__ == "__main__":
    main()
