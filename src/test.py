import tqdm
from src.avgmeter import AverageMeter
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
##define test function
from sklearn.metrics import precision_score, recall_score
import numpy as np
from tqdm.notebook import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.loggers import Logger
from sklearn.metrics import f1_score

logger = Logger()
writer = logger.create_summary_writer()
def calc_f1_score(y_true, y_pred):
    f1 = f1_score(y_true, y_pred)
    labels = ['F1 Score']
    values = [f1]
    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score Comparison')
    plt.tight_layout()
    writer.add_figure('Confusion Matrix', fig)




    # # Calculate the F1 score for each class.
  # f1_scores = []
  # for i in range(len(np.unique(y_true))):
  #   precision = np.true_divide(np.sum(y_true[y_true == i] & y_pred[y_true == i]), np.sum(y_pred[y_true == i]))
  #   recall = np.true_divide(np.sum(y_true[y_true == i] & y_pred[y_true == i]), np.sum(y_true[y_true == i]))
  #   f1_scores.append(2 * (precision * recall) / (precision + recall))
  #
  # # Calculate the average F1 score.
  # f1 = np.mean(f1_scores)



def test(model, dataloader, device):
    # meters
    loss_meter = AverageMeter()
    top1_acc_meter = AverageMeter()
    top5_acc_meter = AverageMeter()
    # switch to test mode
    correct_top1 = 0
    correct_top5 = 0
    y_true = []
    y_pred = []
    model.eval()
    tk = tqdm(dataloader, total=int(len(dataloader)), desc='Test', unit='frames', leave=False)
    for batch_idx, data in enumerate(tk):
        # fetch the data
        frame, label = data[0], data[1]
        y_true.extend(label.cpu().numpy())
        # after fetching the data transfer the model to the 
        # required device, in this example the device is gpu
        # transfer to gpu can also be done by 
        frame, label = frame.to(device), label.to(device)
        # since we dont need to backpropagate loss in testing,
        # we dont keep the gradient
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

        y_pred.extend(pred.cpu().numpy())

        # update the loss meter 
        loss_meter.update(loss_this.item(), label.shape[0])
    f1_score(y_true, y_pred, average='macro')
    print('F1 score Generated')
    print('Test: Average loss: {:.4f}, Top-1 Accuracy: {}/{} ({:.2f}%), Top-5 Accuracy: {}/{} ({:.2f}%)\n'.format(
        loss_meter.avg, correct_top1, len(dataloader.dataset), top1_acc_meter.avg, correct_top5,
        len(dataloader.dataset), top5_acc_meter.avg))
    print('Confusion Matrix:')
    cm = (confusion_matrix(y_true, y_pred))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax)
    plt.tight_layout()
    writer.add_figure('Confusion Matrix', fig)
