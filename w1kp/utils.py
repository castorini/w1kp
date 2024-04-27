__all__ = ['set_seed', 'cached_load_image']

from functools import lru_cache
import random
from pathlib import Path

import PIL.Image
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


@lru_cache(maxsize=4096)
def cached_load_image(path: str) -> PIL.Image.Image:
    with PIL.Image.open(str(path)) as img:
        img.load()
        return img
