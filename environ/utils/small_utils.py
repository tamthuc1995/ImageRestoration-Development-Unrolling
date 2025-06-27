import numpy as np
import os
import random
import time
import torch
from os import path


def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())

def set_random_seed(seed=2204):
    """Set random seeds for reproducible results - Need further check"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pretty_strdict(input_dict, indent_level=0):
    
    msg = '\n'
    for k, v in input_dict.items():
        if isinstance(v, dict):
            msg += '.' * (indent_level * 2) + k + ':{'
            msg += pretty_strdict(v, indent_level + 1)
            msg += '.' * (indent_level * 2) + '}\n'
        else:
            msg += '.' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg
