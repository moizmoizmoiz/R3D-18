
import random
import numpy as np
import torch


def set_random_seed(seed = '2222'):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)