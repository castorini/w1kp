__all__ = ['set_seed']

import random

import numpy as np


def set_seed(seed: int):
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    random.seed(seed)
    np.random.seed(seed)
