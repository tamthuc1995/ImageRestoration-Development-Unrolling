import os, sys 
import itertools
import collections
import random
import time 
import logging

from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pyplot import imread

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.optim import Adam, AdamW

#########################################################################################################
torch.set_float32_matmul_precision('high')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ROOT_PROJECT = "/home/jovyan/shared/Thuc/hoodsgatedrive/projects/ImageRestoration-Development-Unrolling/"
# ROOT_DATASET = "/home/jovyan/shared/Thuc/hoodsgatedrive/projects/"

ROOT_PROJECT = "/home/dotamthuc/Works/Projects/ImageRestoration-Development-Unrolling/"
ROOT_DATASET = "/home/dotamthuc/Works/Projects/ImageRestoration-Development-Unrolling"


#########################################################################################################

sys.path.append(os.path.join(ROOT_PROJECT, 'exploration/model_multiscale_mixture_GLR/lib'))
from dataloader import ImageSuperResolution
import baselineNullRestormer as model_structure


LOG_DIR = os.path.join(ROOT_PROJECT, "exploration/model_multiscale_mixture_GLR/result/model_test_nullrestormer/logs/")
LOGGER = logging.getLogger("main")
logging.basicConfig(
    format='%(asctime)s: %(message)s', 
    datefmt='%m/%d/%Y %I:%M:%S %p',
    filename=os.path.join(LOG_DIR, 'training00.log'), 
    level=logging.INFO
)

CHECKPOINT_DIR = os.path.join(ROOT_PROJECT, "exploration/model_multiscale_mixture_GLR/result/model_test_nullrestormer/checkpoints/")
VERBOSE_RATE = 1000

(H_train, W_train) = (128, 128)
(H_val, W_val) = (128, 128)
(H_test, W_test) = (496, 496)

train_dataset = ImageSuperResolution(
    csv_path=os.path.join(ROOT_DATASET, "dataset/DFWB_training_data_info.csv"),
    dist_mode="vary_addictive_noise",
    lambda_noise=[[1.0, 10.0, 15.0, 20.0, 25.0], [0.1, 0.1, 0.1, 0.1, 0.6]],
    patch_size=(H_train, W_train),
    patch_overlap_size=(H_train//2, W_train//2),
    max_num_patchs=1000000,
    root_folder=ROOT_DATASET,
    logger=LOGGER,
    device=torch.device("cpu"),
)

validation_dataset = ImageSuperResolution(
    csv_path=os.path.join(ROOT_DATASET, "dataset/CBSD68_testing_data_info.csv"),
    dist_mode="addictive_noise",
    lambda_noise=25.0,
    patch_size=(H_val, H_val),
    patch_overlap_size=(H_val//2, H_val//2),
    max_num_patchs=1000000,
    root_folder=ROOT_DATASET,
    logger=LOGGER,
    device=torch.device("cpu"),
)

test_dataset = ImageSuperResolution(
    csv_path=os.path.join(ROOT_DATASET, "dataset/McMaster_testing_data_info.csv"),
    dist_mode="addictive_noise",
    lambda_noise=25.0,
    patch_size=(H_test, W_test),
    patch_overlap_size=(0, 0),
    max_num_patchs=1000000,
    root_folder=ROOT_DATASET,
    logger=LOGGER,
    device=torch.device("cpu"),
)


data_train_batched = torch.utils.data.DataLoader(
    train_dataset, batch_size=4, num_workers=4
)

data_valid_batched = torch.utils.data.DataLoader(
    validation_dataset, batch_size=16, num_workers=4
)

data_test_batched = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, num_workers=4
)

NUM_EPOCHS = 45

CONNECTION_FLAGS = np.array([
    1,1,1,
    1,0,1,
    1,1,1,
]).reshape((3,3))


model = model_structure.Restormer(**{
    "inp_channels":3, 
    "out_channels":3, 
    "dim": 48,
    "num_blocks": [2,3,3,4], 
    "num_refinement_blocks": 4,
    "heads": [1,2,4,8],
    "ffn_expansion_factor": 1.0,
    "bias": False,
    "LayerNorm_type": 'WithBias',   ## Other option 'BiasFree'
    "dual_pixel_task": False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
}).to(DEVICE)


s = 0
for p in model.parameters():
    s += np.prod(np.array(p.shape))
    # print(p.dtype, np.array(p.shape), s)

LOGGER.info(f"Init model with total parameters: {s}")

criterian = nn.L1Loss()
optimizer = AdamW(
    model.parameters(),
    lr=0.0003,
    eps=1e-08
)

### TRAINING
LOGGER.info("######################################################################################")
LOGGER.info("BEGIN TRAINING PROCESS")
# training_state_path = os.path.join(CHECKPOINT_DIR, 'checkpoints_epoch00_iter0094k.pt')
# training_state = torch.load(training_state_path)
# model.load_state_dict(training_state["model"])
# optimizer.load_state_dict(training_state["optimizer"])
# i_checkpoint=training_state["i"]


for epoch in range(NUM_EPOCHS):

    model.train()

    i = 0
    ### TRAINING
    list_train_mse = []
    list_train_psnr = []
    for patchs_noisy, patchs_true in data_train_batched:
        s = time.time()
        optimizer.zero_grad()
        patchs_noisy = patchs_noisy.to(DEVICE)
        patchs_true = patchs_true.to(DEVICE) 
        reconstruct_patchs = model(patchs_noisy.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        loss_value = criterian(reconstruct_patchs, patchs_true)
        loss_value.backward()
        optimizer.step()

        img_true = np.clip(patchs_true.detach().cpu().numpy(), a_min=0.0, a_max=1.0).astype(np.float64)
        img_recon = np.clip(reconstruct_patchs.detach().cpu().numpy(), a_min=0.0, a_max=1.0).astype(np.float64)
        train_mse_value = np.square(img_true- img_recon).mean()
        train_psnr = 10 * np.log10(1/train_mse_value)
        # LOGGER.info(f"iter={i} time={time.time()-s} psnr={train_psnr} mse={train_mse_value}")
        list_train_psnr.append(train_psnr)
        list_train_mse.append(train_mse_value)


        if (i%(VERBOSE_RATE//10) == 0):
            LOGGER.info(f"iter={i} time={time.time()-s} psnr={np.mean(list_train_psnr[-100:])} mse={np.mean(list_train_mse[-100:])}")
            list_train_mse  = list_train_mse[-100:].copy()
            list_train_psnr = list_train_psnr[-100:].copy()


        if (i%VERBOSE_RATE == 0):
            checkpoint = { 
                'i': i,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, f'checkpoints_epoch{str(epoch).zfill(2)}_iter{str(i//VERBOSE_RATE).zfill(4)}k.pt'))


        if (i%(VERBOSE_RATE//2) == 0):
            # LOGGER.info(f"Start VALIDATION EPOCH {epoch} - iter={i}")
            # model.graph_frame_recalibrate(H_val, W_val)

            # ### VALIDAING
            model.eval()
            list_val_mse = []
            val_i = 0
            for val_patchs_noisy, val_patchs_true in data_valid_batched:
                s = time.time()
                with torch.no_grad():
                    val_patchs_noisy = val_patchs_noisy.to(DEVICE)
                    val_patchs_true = val_patchs_true.to(DEVICE) 

                    reconstruct_patchs = model(val_patchs_noisy.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                    img_true = np.clip(val_patchs_true.cpu().numpy(), a_min=0.0, a_max=1.0).astype(np.float64)
                    img_recon = np.clip(reconstruct_patchs.cpu().numpy(), a_min=0.0, a_max=1.0).astype(np.float64)
                    val_mse_value = np.square(img_true- img_recon).mean()
                    list_val_mse.append(val_mse_value)
                    # LOGGER.info(f"test_i={test_i} time={time.time()-s} test_i_psnr_value={10 * np.log10(1/test_mse_value)}")
                val_i+=1

            psnr_validation = 10 * np.log10(1/np.array(list_val_mse))
            LOGGER.info(f"FINISH VALIDATION EPOCH {epoch} - iter={i} -  psnr_validation={np.mean(psnr_validation)}")
            # model.graph_frame_recalibrate(H_train, W_train)
            model.train()

        if (i%VERBOSE_RATE == 0):

            # LOGGER.info(f"Start VALIDATION EPOCH {epoch} - iter={i}")
            # model.graph_frame_recalibrate(H_test, W_test)

            # ### VALIDAING
            model.eval()
            list_test_mse = []
            test_i = 0
            for test_patchs_noisy, test_patchs_true in data_test_batched:
                s = time.time()
                with torch.no_grad():
                    test_patchs_noisy = test_patchs_noisy.to(DEVICE)
                    test_patchs_true = test_patchs_true.to(DEVICE) 
                    reconstruct_patchs = model(test_patchs_noisy.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                    img_true = np.clip(test_patchs_true[0].cpu().numpy(), a_min=0.0, a_max=1.0).astype(np.float64)
                    img_recon = np.clip(reconstruct_patchs[0].cpu().numpy(), a_min=0.0, a_max=1.0).astype(np.float64)
                    test_mse_value = np.square(img_true- img_recon).mean()
                    list_test_mse.append(test_mse_value)
                    # LOGGER.info(f"test_i={test_i} time={time.time()-s} test_i_psnr_value={10 * np.log10(1/test_mse_value)}")
                test_i+=1

            psnr_testing = 10 * np.log10(1/np.array(list_test_mse))
            LOGGER.info(f"FINISH TESING EPOCH {epoch} - iter={i} -  psnr_testing={np.mean(psnr_testing)}")
            # model.graph_frame_recalibrate(H_train, W_train)
            model.train()

        i+=1     