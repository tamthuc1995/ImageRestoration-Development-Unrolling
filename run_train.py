import argparse
import datetime
import logging
import math
import random
import time
import torch
import os
import numpy as np


from environ.utils.small_utils import set_random_seed, get_time_str, pretty_strdict
from environ.utils.custom_parser import parse
from environ.utils.custom_logger import get_root_logger

from environ.data import create_dataset, create_dataloader
from environ.data.data_sampler import ResumeableSampler


def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-yaml_path', type=str, required=True, help='Path to option YAML file.')
    
    args = parser.parse_args()
    environ_conf = parse(args.yaml_path, is_train=is_train)

    # random seed
    seed = environ_conf.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        environ_conf['manual_seed'] = seed
    set_random_seed(seed)

    return environ_conf


def main():
    # parse options, set distributed setting, set ramdom seed
    environ_conf = parse_options(is_train=True)
    ROOT_PATH = environ_conf['path']["root_dir"]

    # automatic resume running experiment from latest checkpoint
    checkpoints_folder_path = 'experiments/{}/learning_checkpoints/'.format(environ_conf['name'])
    checkpoints_folder_path = os.path.join(ROOT_PATH, checkpoints_folder_path)
    try:
        checkpoints = sorted(os.listdir(checkpoints_folder_path))
    except:
        checkpoints = []

    latest_checkpoint = None
    if len(checkpoints) > 0:
        max_state_file = checkpoints[-1]
        latest_checkpoint_path = os.path.join(checkpoints_folder_path, max_state_file)
        environ_conf['path']['latest_checkpoint_path'] = latest_checkpoint_path

    # load latest checkpoint states if necessary
    if environ_conf['path'].get('latest_checkpoint_path'):
        latest_checkpoint = torch.load(environ_conf['path']['latest_checkpoint_path'])
    else:
        latest_checkpoint = None

    # mkdir for experiments checkpoints and logging
    if latest_checkpoint is None:
        ## Create checkpoints folder
        os.makedirs(checkpoints_folder_path, exist_ok=True)
        environ_conf["path"]["checkpoints_folder"] = checkpoints_folder_path

        ## Create folder for logging files
        logs_folder_path = 'experiments/{}/log_files/'.format(environ_conf['name'])
        logs_folder_path = os.path.join(ROOT_PATH, logs_folder_path)
        os.makedirs(logs_folder_path, exist_ok=True)
        environ_conf["path"]["log_files_folder"] = logs_folder_path


    # initialize loggers
    log_file = os.path.join(
        environ_conf['path']['log_files_folder'], 
        f"run_train_{environ_conf['name']}.log"
    )
    logger = get_root_logger(
        logger_name=environ_conf["name"],
        log_level=logging.INFO,
        log_file=log_file
    )
    logger.info(f"environ_conf={pretty_strdict(environ_conf)}")

    # Create dataloader for training and validation
    train_dataset_conf = environ_conf['datasets']['train']
    train_dataset = create_dataset(train_dataset_conf, environ_conf)
    train_sampler = ResumeableSampler(train_dataset)
    train_dataloader = create_dataloader(train_dataset, train_sampler, train_dataset_conf, environ_conf)

    # train_sampler.set_epoch_and_current_sample(
    #     current_epoch=10,
    #     current_sample=200
    # )

    return None




if __name__ == '__main__':
    main()



