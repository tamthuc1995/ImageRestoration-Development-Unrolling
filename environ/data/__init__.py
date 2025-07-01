import importlib
import numpy as np
import random
import torch
import torch.utils.data
from functools import partial
import os
import glob

from environ.utils.custom_logger import get_root_logger



__all__ = ['create_dataset']

# automatically scan and import dataset modules
# scan all the files under the data folder with '_dataset' in file names
data_folder = os.path.dirname(os.path.abspath(__file__))
dataset_filenames = [
    os.path.splitext(os.path.basename(v))[0] for v in glob.glob(os.path.join(data_folder, "*_dataset.py"))
]
# import all the dataset modules
_dataset_modules = [
    importlib.import_module(f'environ.data.{file_name}')
    for file_name in dataset_filenames
]


def create_dataset(dataset_conf, environ_conf):
    """
    A simple dynamic class creator
    """
    dataset_type = dataset_conf['type']
    
    # dynamic instantiation
    for module in _dataset_modules:
        dataset_cls = getattr(module, dataset_type, None)
        if dataset_cls is not None:
            break
    if dataset_cls is None:
        raise ValueError(f'Dataset {dataset_type} is not found.')

    dataset = dataset_cls(**dataset_conf["dataset_args"])

    logger = get_root_logger(environ_conf["name"])
    logger.info(
        f'Dataset {dataset.__class__.__name__} - {dataset_conf["name"]} is created.'
    )
    
    return dataset


def create_dataloader(dataset, sampler, dataset_conf, environ_conf):
    """
    Create dataloader.
    """
    
    dataloader_args = dataset_conf["dataloader_args"]
    dataloader_args["dataset"] = dataset
    dataloader_args["sampler"] = sampler

    logger = get_root_logger(environ_conf["name"])
    logger.info(
        f'Dataloader from {dataset.__class__.__name__} - {dataset_conf["name"]} is created.'
    )
    dataloader = torch.utils.data.DataLoader(**dataloader_args)
    

    return dataloader