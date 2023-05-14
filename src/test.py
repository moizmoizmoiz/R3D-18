import tqdm
from src.avgmeter import AverageMeter
import torch
import torch.nn.functional as F


def test(model, dataloader, device):
    # meters
    loss_meter = AverageMeter()
    top1_acc_meter = AverageMeter()
    top5_acc_meter = AverageMeter()
    # switch to test mode
    correct_top1 = 0
    correct_top5 = 0
    model.eval()
    tk = tqdm(dataloader, total=int(len(dataloader)), desc='Test', unit='frames', leave=False)
    for batch_idx, data in enumerate(tk):
        # fetch the data
        frame, label = data[0], data[1]
        # after fetching the data transfer the model to the
        # required device, in this example the device is gpu
        # transfer to gpu can also be done by
        frame, label = frame.to(device), label.to(device)
        # since we don't need to back propagate loss in testing,
        # we don't keep the gradient
        with torch.no_grad():
            output = model(frame)
        # compute the loss function just for checking
        loss_this = F.cross_entropy(output, label)
        # get the index of the max log-probability
        pred = output.argmax(dim=1, keepdim=True)
        # check which of the predictions are correct
        correct_this_top1 = pred.eq(label.view_as(pred)).sum().item()
        # accumulate the correct ones for top-1 accuracy
        correct_top1 += correct_this_top1
        # compute top-1 accuracy
        acc_this_top1 = correct_this_top1 / label.shape[0] * 100.0
        # update the top-1 accuracy meter
        top1_acc_meter.update(acc_this_top1, label.shape[0])

        # compute top-5 accuracy
        _, pred_top5 = torch.topk(output, k=5, dim=1)
        pred_top5 = pred_top5.t()
        correct_this_top5 = pred_top5.eq(label.view(1, -1).expand_as(pred_top5)).sum().item()
        correct_top5 += correct_this_top5
        acc_this_top5 = correct_this_top5 / label.shape[0] * 100.0
        top5_acc_meter.update(acc_this_top5, label.shape[0])

        # update the loss meter
        loss_meter.update(loss_this.item(), label.shape[0])
    print('Test: Average loss: {:.4f}, Top-1 Accuracy: {}/{} ({:.2f}%), Top-5 Accuracy: {}/{} ({:.2f}%)\n'.format(
        loss_meter.avg, correct_top1, len(dataloader.dataset), top1_acc_meter.avg, correct_top5,
        len(dataloader.dataset), top5_acc_meter.avg))


def get_confusion_matrix(pred, label, num_classes):
    conf_mat = torch.zeros(num_classes, num_classes)
    for i in range(len(pred)):
        actual_class = int(label[i])
        predicted = int(pred[i])
        conf_mat[actual_class][predicted] += 1
    return conf_mat