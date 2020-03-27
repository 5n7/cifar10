import random

import numpy as np
import torch

__all__ = ["set_seed"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
