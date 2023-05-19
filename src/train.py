from tqdm.notebook import tqdm
import torch.nn.functional as F
from src.avgmeter import AverageMeter
import torch


##define train function
def train(model, dataloader, optimizer, device):
    # meter
    loss_meter = AverageMeter()
    # switch to train mode
    model.train()
    tk = tqdm(dataloader, total=int(len(dataloader)), desc='Training', unit='frames')
    for batch_idx, data in enumerate(tk):
        # fetch the data
        frame, label = data[0], data[1]
        # after fetching the data, transfer the model to the
        # required device, in this example the device is gpu
        # transfer to gpu can also be done by
        frame, label = frame.to(device), label.to(device)
        # compute the forward pass
        output = model(frame)
        # compute the loss function
        loss_this = F.cross_entropy(output, label)  # TODO variable
        # initialize the optimizer
        optimizer.zero_grad()
        # compute the backward pass
        loss_this.backward()
        # update the parameters
        optimizer.step()
        # update the loss meter
        loss_meter.update(loss_this.item(), label.shape[0])
        tk.set_postfix({"loss": loss_meter.avg})
        avg_loss = loss_meter.avg
    return avg_loss
