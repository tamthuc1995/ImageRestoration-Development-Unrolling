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
from torch.optim import Adam, Adadelta, SGD

#########################################################################################################
# torch.autograd.set_detect_anomaly(True)
torch.set_float32_matmul_precision('high')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ROOT = "/home/dotamthuc/Works/Projects/"
LOG_DIR = os.path.join(ROOT, "unrollGTV/model_multiscale_mixture_GLR/result/model_test09/logs/")
sys.path.append(os.path.join(ROOT, 'unrollGTV/model_multiscale_mixture_GLR/lib'))
LOGGER = logging.getLogger("main")
logging.basicConfig(
    format='%(asctime)s: %(message)s', 
    datefmt='%m/%d/%Y %I:%M:%S %p',
    filename=os.path.join(LOG_DIR, 'training00.log'), 
    level=logging.INFO
)
#########################################################################################################

from dataloader import ImageSuperResolution
import model_MMGLR as model_structure


CHECKPOINT_DIR = os.path.join(ROOT, "unrollGTV/model_multiscale_mixture_GLR/result/model_test09/checkpoints/")
VERBOSE_RATE = 1000
(H_train, W_train) = (128, 128)
(H_val, W_val) = (128, 128)
(H_test, W_test) = (496, 496)

train_dataset = ImageSuperResolution(
    csv_path="/home/dotamthuc/Works/Projects/ImageRestoration-Development-Unrolling/dataset/DFWB_training_data_info.csv",
    dist_mode="addictive_noise_scale",
    lambda_noise=25.0,
    patch_size=H_train,
    patch_overlap_size=H_train//2,
    max_num_patchs=1200000,
    root_folder="/home/dotamthuc/Works/Projects/ImageRestoration-Development-Unrolling/",
    logger=LOGGER,
    device=DEVICE,
)

validation_dataset = ImageSuperResolution(
    csv_path="/home/dotamthuc/Works/Projects/ImageRestoration-Development-Unrolling/dataset/CBSD68_testing_data_info.csv",
    dist_mode="addictive_noise",
    lambda_noise=25.0,
    patch_size=H_val,
    patch_overlap_size=0,
    max_num_patchs=10000,
    root_folder="/home/dotamthuc/Works/Projects/ImageRestoration-Development-Unrolling/",
    logger=LOGGER,
    device=DEVICE,
)

test_dataset = ImageSuperResolution(
    csv_path="/home/dotamthuc/Works/Projects/unrollGTV/data/datasets/McM_test_info.csv",
    dist_mode="addictive_noise",
    lambda_noise=25.0,
    patch_size=H_test,
    patch_overlap_size=0,
    max_num_patchs=10000,
    root_folder="/home/dotamthuc/Works/Projects/unrollGTV/data",
    logger=LOGGER,
    device=DEVICE,
)


data_train_batched = torch.utils.data.DataLoader(
    train_dataset, batch_size=4
)

data_valid_batched = torch.utils.data.DataLoader(
    validation_dataset, batch_size=16
)

data_test_batched = torch.utils.data.DataLoader(
    test_dataset, batch_size=1
)

NUM_EPOCHS = 45
# CONNECTION_FLAGS = np.array([
#     1,1,1,1,1,
#     1,1,1,1,1,
#     1,1,0,1,1,
#     1,1,1,1,1,
#     1,1,1,1,1,
# ]).reshape((5,5))
# CONNECTION_FLAGS = np.array([
#     1,0,1,0,1,
#     0,1,1,1,0,
#     1,1,0,1,1,
#     0,1,1,1,0,
#     1,0,1,0,1,
# ]).reshape((5,5))

CONNECTION_FLAGS = np.array([
    1,1,1,
    1,0,1,
    1,1,1,
]).reshape((3,3))

modelConf = {"ModelLightWeightTransformerGLR": {
    "img_height":H_train,
    "img_width":W_train,
    "n_blocks":5,
    "n_graphs":9,
    "n_levels":4,
    "device": DEVICE,
    "global_mmglr_confs" : {
        "n_graphs":9,
        "n_levels":4,
        "n_cgd_iters":5,
        "alpha_init":0.5,
        "muy_init": torch.tensor([[0.3], [0.15], [0.075], [0.0375]]).to(DEVICE),
        "beta_init":0.1,
        "device":DEVICE,
        "GLR_modules_conf": [
            {"GLRConf":{
                "input_width": H_train,
                "input_height": W_train,
                "n_node_fts": 8,
                "n_graphs": 9,
                "connection_window": CONNECTION_FLAGS,
                "device": DEVICE,
                "M_diag_init": 1.0,
                "Mxy_diag_init": 1.0
            }},
            {"GLRConf":{
                "input_width": H_train//2,
                "input_height": W_train//2,
                "n_node_fts": 8,
                "n_graphs": 9,
                "connection_window": CONNECTION_FLAGS,
                "device": DEVICE,
                "M_diag_init": 1.0,
                "Mxy_diag_init": 1.0
            }},
            {"GLRConf":{
                "input_width": H_train//4,
                "input_height": W_train//4,
                "n_node_fts": 16,
                "n_graphs": 9,
                "connection_window": CONNECTION_FLAGS,
                "device": DEVICE,
                "M_diag_init": 1.0,
                "Mxy_diag_init": 1.0
            }},
            {"GLRConf":{
                "input_width": H_train//8,
                "input_height": W_train//8,
                "n_node_fts": 32,
                "n_graphs": 9,
                "connection_window": CONNECTION_FLAGS,
                "device": DEVICE,
                "M_diag_init": 1.0,
                "Mxy_diag_init": 1.0
            }},
        ],
        "Extractor_modules_conf":[
            {"ExtractorConf":{
                "n_features_in": 72,
                "n_features_out": 72,
                "n_channels_in": 3,
                "n_channels_out": 3,
                "device": DEVICE
            }},
            {"ExtractorConf":{
                "n_features_in": 72,
                "n_features_out": 144,
                "n_channels_in": 3,
                "n_channels_out": 3,
                "device": DEVICE
            }},
            {"ExtractorConf":{
                "n_features_in": 144,
                "n_features_out": 288,
                "n_channels_in": 3,
                "n_channels_out": 3,
                "device": DEVICE
            }},
        ]}
    }
}

model = model_structure.ModelLightWeightTransformerGLR(**modelConf["ModelLightWeightTransformerGLR"])
# model.compile()


s = 0
for p in model.parameters():
    s += np.prod(np.array(p.shape))
    # print(p.dtype, np.array(p.shape), s)

LOGGER.info(f"Init model with total parameters: {s}")


# model = torch.compile(model)
criterian = nn.MSELoss()
optimizer = Adam(
    model.parameters(),
    lr=0.001,
    eps=1e-08
)

### TRAINING
LOGGER.info("######################################################################################")
LOGGER.info("BEGIN TRAINING PROCESS")
training_state_path = os.path.join(CHECKPOINT_DIR, 'checkpoints_epoch00_iter0024k.pt')
training_state = torch.load(training_state_path)
model.load_state_dict(training_state["model"])
optimizer.load_state_dict(training_state["optimizer"])
i_checkpoint=training_state["i"]


for epoch in range(NUM_EPOCHS):

    model.train()

    i = 0
    ### TRAINING
    for patchs_noisy, patchs_true in data_train_batched:
        # if i <= i_checkpoint:
        #     print(f"iter={i}")
        #     i+=1
        #     continue
        
        s = time.time()
        optimizer.zero_grad()
        
        reconstruct_patchs = model(patchs_noisy)
        loss_mse = criterian(reconstruct_patchs, patchs_true)
        loss_mse.backward()
        optimizer.step()

        mse_value = loss_mse.detach().cpu().numpy()
        psnr = 10 * np.log10(1/mse_value)
        LOGGER.info(f"iter={i} time={time.time()-s} psnr={psnr} mse={mse_value}")

        if (i%(VERBOSE_RATE/5) == 0):
            # LOGGER.info(f"Start VALIDATION EPOCH {epoch} - iter={i}")
            model.graph_frame_recalibrate(H_val, W_val)

            # ### VALIDAING
            model.eval()
            list_val_mse = []
            val_i = 0
            for val_patchs_noisy, val_patchs_true in data_valid_batched:
                s = time.time()
                with torch.no_grad():
                    reconstruct_patchs = model(val_patchs_noisy)
                    img_true = np.clip(val_patchs_true.cpu().numpy(), a_min=0.0, a_max=1.0).astype(np.float64)
                    img_recon = np.clip(reconstruct_patchs.cpu().numpy(), a_min=0.0, a_max=1.0).astype(np.float64)
                    val_mse_value = np.square(img_true- img_recon).mean()
                    list_val_mse.append(val_mse_value)
                    # LOGGER.info(f"test_i={test_i} time={time.time()-s} test_i_psnr_value={10 * np.log10(1/test_mse_value)}")
                val_i+=1

            psnr_validation = 10 * np.log10(1/np.array(list_val_mse))
            LOGGER.info(f"FINISH VALIDATION EPOCH {epoch} - iter={i} -  psnr_validation={np.mean(psnr_validation)}")
            model.graph_frame_recalibrate(H_train, W_train)
            model.train()


        if (i%VERBOSE_RATE == 0):
            # torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f'test_{str(i).zfill(4)}.pt'))
            checkpoint = { 
                'i': i,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, f'checkpoints_epoch{str(epoch).zfill(2)}_iter{str(i//VERBOSE_RATE).zfill(4)}k.pt'))

        if (i%VERBOSE_RATE == 0):

            # LOGGER.info(f"Start VALIDATION EPOCH {epoch} - iter={i}")
            model.graph_frame_recalibrate(H_test, W_test)

            # ### VALIDAING
            model.eval()
            list_test_mse = []
            test_i = 0
            for test_patchs_noisy, test_patchs_true in data_test_batched:
                s = time.time()
                with torch.no_grad():
                    reconstruct_patchs = model(test_patchs_noisy)

                    img_true = np.clip(test_patchs_true[0].cpu().numpy(), a_min=0.0, a_max=1.0).astype(np.float64)
                    img_recon = np.clip(reconstruct_patchs[0].cpu().numpy(), a_min=0.0, a_max=1.0).astype(np.float64)
                    test_mse_value = np.square(img_true- img_recon).mean()
                    list_test_mse.append(test_mse_value)
                    # LOGGER.info(f"test_i={test_i} time={time.time()-s} test_i_psnr_value={10 * np.log10(1/test_mse_value)}")
                test_i+=1

            psnr_testing = 10 * np.log10(1/np.array(list_test_mse))
            LOGGER.info(f"FINISH TESING EPOCH {epoch} - iter={i} -  psnr_testing={np.mean(psnr_testing)}")
            model.graph_frame_recalibrate(H_train, W_train)
            model.train()

        i+=1     