__all__ = ['set_seed', 'load_image', 'apply_ema', 'compute_cdf', 'query_inverted_cdf']

from functools import lru_cache
import random
from pathlib import Path

import PIL.Image
import numpy as np
import torch
from torch import nn


def set_seed(seed: int):
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    random.seed(seed)
    np.random.seed(seed)


def load_image(path: str) -> PIL.Image.Image:
    with PIL.Image.open(str(path)) as img:
        img.load()
        return img


EMA_WEIGHTS = {}


@torch.no_grad()
def apply_ema(model: nn.Module, decay: float = 0.95):
    """Applies exponential weight averaging to the model parameters in `model`. Higher `decay` values result in higher
    weights on the past iterate."""
    global EMA_WEIGHTS

    for name, param in model.named_parameters():
        if param.requires_grad and name not in EMA_WEIGHTS:
            EMA_WEIGHTS[name] = param.data.clone()

        if param.requires_grad:
            param.data.mul_(1 - decay).add_(EMA_WEIGHTS[name], alpha=decay)
            EMA_WEIGHTS[name].copy_(param.data)


def compute_cdf(data: np.ndarray, left=0, right=1):
    data = data.reshape(-1)
    data = np.hstack([np.array([left]), data, np.array([right])])
    cdf_x = np.sort(data)
    cdf_y = np.arange(len(data)) / len(data)

    return cdf_x, cdf_y


def query_inverted_cdf(cdf_x: np.ndarray, cdf_y: np.ndarray, queries: np.ndarray) -> np.ndarray:
    return np.interp(queries, cdf_x, cdf_y)
