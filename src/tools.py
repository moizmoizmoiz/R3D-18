import os
import os.path as osp
import warnings
import errno
import torch.nn as nn


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def check_isfile(path):
    isfile = osp.isfile(path)
    if not isfile:
        warnings.warn(f'No file found at "{path}"')
    return isfile


def count_num_param(model):
    num_param = sum(p.numel() for p in model.parameters()) / 1e06

    if isinstance(model, nn.DataParallel):
        model = model.module

    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Module):
        # we ignore the classifier because it is unused at test time
        num_param -= sum(p.numel() for p in model.classifier.parameters()) / 1e06
    return num_param
