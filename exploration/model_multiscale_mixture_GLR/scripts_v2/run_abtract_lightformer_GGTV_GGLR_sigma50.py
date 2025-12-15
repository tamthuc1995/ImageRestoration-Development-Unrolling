import os, sys 
import itertools
import collections
import random
import time 
import logging

from PIL import Image
from skimage import img_as_ubyte
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pyplot import imread

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, SequentialLR

#########################################################################################################
torch.set_float32_matmul_precision('high')
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEVICE)
# torch.autograd.set_detect_anomaly(True)

ROOT_PROJECT = "/home/jovyan/shared/Thuc/hoodsgatedrive/projects/ImageRestoration-Development-Unrolling/"
ROOT_DATASET = "/home/jovyan/shared/Thuc/hoodsgatedrive/projects/"

#########################################################################################################

sys.path.append(os.path.join(ROOT_PROJECT, 'exploration/model_multiscale_mixture_GLR/lib'))
from dataloader_v2 import ImageSuperResolution
import model_GLR_GTV_deep_v21 as model_structure


LOG_DIR = os.path.join(ROOT_PROJECT, "exploration/model_multiscale_mixture_GLR/result_v2/model_v20_sigma_50/logs/")
LOGGER = logging.getLogger("main")
logging.basicConfig(
    format='%(asctime)s: %(message)s', 
    datefmt='%m/%d/%Y %I:%M:%S %p',
    filename=os.path.join(LOG_DIR, 'training00.log'), 
    level=logging.INFO
)

CHECKPOINT_DIR = os.path.join(ROOT_PROJECT, "exploration/model_multiscale_mixture_GLR/result_v2/model_v20_sigma_50/checkpoints/")
VERBOSE_RATE = 1000

(H_train01, W_train01) = (128, 128)
(H_train02, W_train02) = (192, 192)
(H_train03, W_train03) = (256, 256)
(H_train04, W_train04) = (384, 384)

train_dataset01 = ImageSuperResolution(
    csv_path=os.path.join(ROOT_DATASET, "dataset/DFWB_training_data_info.csv"),
    dist_mode="addictive_noise_scale",
    lambda_noise=50.0,
    use_data_aug=True,
    patch_size=(H_train01,H_train01),
    max_num_patchs=800000,
    root_folder=ROOT_DATASET,
    logger=LOGGER,
    device=torch.device("cpu"),
)
data_train_batched01 = torch.utils.data.DataLoader(
    train_dataset01, batch_size=4, num_workers=4
)

train_dataset02 = ImageSuperResolution(
    csv_path=os.path.join(ROOT_DATASET, "dataset/DFWB_training_data_info.csv"),
    dist_mode="addictive_noise_scale",
    lambda_noise=50.0,
    use_data_aug=True,
    patch_size=(H_train02,H_train02),
    max_num_patchs=600000,
    root_folder=ROOT_DATASET,
    logger=LOGGER,
    device=torch.device("cpu"),
    seed=2224
)
data_train_batched02 = torch.utils.data.DataLoader(
    train_dataset02, batch_size=3, num_workers=4
)

train_dataset03 = ImageSuperResolution(
    csv_path=os.path.join(ROOT_DATASET, "dataset/DFWB_training_data_info.csv"),
    dist_mode="addictive_noise_scale",
    lambda_noise=50.0,
    use_data_aug=True,
    patch_size=(H_train03,H_train03),
    max_num_patchs=400000,
    root_folder=ROOT_DATASET,
    logger=LOGGER,
    device=torch.device("cpu"),
)
data_train_batched03 = torch.utils.data.DataLoader(
    train_dataset03, batch_size=2, num_workers=4
)

train_dataset04 = ImageSuperResolution(
    csv_path=os.path.join(ROOT_DATASET, "dataset/DFWB_training_data_info.csv"),
    dist_mode="addictive_noise_scale",
    lambda_noise=50.0,
    use_data_aug=True,
    patch_size=(H_train04,H_train04),
    max_num_patchs=200000,
    root_folder=ROOT_DATASET,
    logger=LOGGER,
    device=torch.device("cpu"),
)

data_train_batched04 = torch.utils.data.DataLoader(
    train_dataset04, batch_size=1, num_workers=4
)


NUM_EPOCHS = 1

model = model_structure.AbtractMultiScaleGraphFilter(
    n_channels_in=3, 
    n_channels_out=3, 
    dims=[48, 96, 192, 384],
    hidden_dims=[96, 192, 384, 768],
    nsubnets=[1, 1, 1, 1],
    ngraphs=[8, 16, 16, 32], 
    num_blocks=[4, 6, 6, 8], 
    num_blocks_out=4
).to(DEVICE)
model.compile()

s = 0
for p in model.parameters():
    s += np.prod(np.array(p.shape))
    # print(p.dtype, np.array(p.shape), s)

LOGGER.info(f"Init model with total parameters: {s}")

criterian01 = nn.L1Loss()
criterian02 = nn.MSELoss()
loss02_weight = 0.1

optimizer = Adam(
    model.parameters(),
    lr=0.0004,
    eps=1e-08
)
lr_scheduler01 = MultiStepLR(
    optimizer,
    milestones=[50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000, 550000, 600000],
    gamma=np.sqrt(np.sqrt(0.5))
)
lr_scheduler02 = CosineAnnealingLR(optimizer, T_max=701000, eta_min=0.000001)
lr_scheduler02.base_lrs = [0.00005 for group in optimizer.param_groups]

lr_scheduler = SequentialLR(
    optimizer,
    schedulers=[lr_scheduler01, lr_scheduler02],
    milestones=[600000],
)


### TRAINING
LOGGER.info("######################################################################################")
LOGGER.info("BEGIN TRAINING PROCESS")
# training_state_path = os.path.join(CHECKPOINT_DIR, 'checkpoints_epoch00_iter0330k.pt')
# training_state = torch.load(training_state_path, weights_only=False)
# model.load_state_dict(training_state["model"])
# optimizer.load_state_dict(training_state["optimizer"])
# lr_scheduler.load_state_dict(training_state["lr_scheduler"])
# i=training_state["i"]
i = 0


for epoch in range(NUM_EPOCHS):

    model.train()

    ### TRAINING
    list_train_mse = []
    list_train_psnr = []
    combined_dataloader = itertools.chain(data_train_batched01, data_train_batched02, data_train_batched03, data_train_batched04)
    for patchs_noisy, patchs_true in combined_dataloader:
        s = time.time()
        optimizer.zero_grad()
        patchs_noisy = patchs_noisy.to(DEVICE)
        patchs_true = patchs_true.to(DEVICE) 
        reconstruct_patchs = model(patchs_noisy.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        reconstruct_patchs_true = model.enc_dec(patchs_true.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        loss_value = criterian01(reconstruct_patchs, patchs_true) + loss02_weight * criterian02(reconstruct_patchs_true, patchs_true)
        loss_value.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
        optimizer.step()
        lr_scheduler.step()

        img_true = np.clip(patchs_true.detach().cpu().numpy(), a_min=0.0, a_max=1.0).astype(np.float64)
        img_recon = np.clip(reconstruct_patchs.detach().cpu().numpy(), a_min=0.0, a_max=1.0).astype(np.float64)
        train_mse_value = np.square(img_true- img_recon).mean()
        train_psnr = 10 * np.log10(1/train_mse_value)
        list_train_psnr.append(train_psnr)
        list_train_mse.append(train_mse_value)


        if (i%(VERBOSE_RATE//10) == 0):
            LOGGER.info(f"iter={i} time={time.time()-s} psnr={np.mean(list_train_psnr[-100:])} mse={np.mean(list_train_mse[-100:])}")
            list_train_mse  = list_train_mse[-100:].copy()
            list_train_psnr = list_train_psnr[-100:].copy()

        if ((i%(5*VERBOSE_RATE) == 0) or ((i >= 690000) and (i%(VERBOSE_RATE) == 0))):
            checkpoint = { 
                'i': i,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, f'checkpoints_epoch{str(epoch).zfill(2)}_iter{str(i//VERBOSE_RATE).zfill(4)}k.pt'))


        if (i%VERBOSE_RATE == 0):
            
            model.eval()
            csv_path = os.path.join(ROOT_DATASET, "dataset/CBSD68_testing_data_info.csv")
            img_infos = pd.read_csv(csv_path, index_col='index')

            paths = img_infos["path"].tolist()
            paths = [
                os.path.join(ROOT_DATASET,path)
                for path in paths
            ]

            sigma_test = 50.0
            factor = 16
            list_test_mse = []
            random_state = np.random.RandomState(seed=2204)
            test_i = 0
            s = time.time()
            for file_ in paths:
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()

                img = Image.open(file_)
                img_true_255 = np.array(img).astype(np.float32)
                img_true = img_true_255 / 255.0

                noisy_img_raw = img_true.copy()
                noisy_img_raw += random_state.normal(0, sigma_test/255., img_true.shape)

                noisy_img = torch.from_numpy(noisy_img_raw).permute(2,0,1)
                noisy_img = noisy_img.unsqueeze(0)

                h,w = noisy_img.shape[2], noisy_img.shape[3]
                H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
                padh = H-h if h%factor!=0 else 0
                padw = W-w if w%factor!=0 else 0
                noisy_img = nn.functional.pad(noisy_img, (0,padw,0,padh), 'reflect')
                
                with torch.no_grad():
                    restored = model(noisy_img.to(DEVICE))

                restored = restored[:,:,:h,:w]
                restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy().copy()

                restored = img_as_ubyte(restored).astype(np.float32)
                test_mse_value = np.square(img_true_255- restored).mean()
                list_test_mse.append(test_mse_value)
                # print(f"test_i={test_i} time={time.time()-s} test_i_psnr_value={20 * np.log10(255.0 / np.sqrt(test_mse_value))}")  
                test_i += 1
                s = time.time()

            psnr_testing = 20 * np.log10(255.0 / np.sqrt(list_test_mse))
            LOGGER.info(f"FINISH VAL EPOCH {epoch} - iter={i} -  psnr_testing={np.mean(psnr_testing)}")
            model.train()

        if (i%VERBOSE_RATE == 0):

            model.eval()
            csv_path = os.path.join(ROOT_DATASET, "dataset/McMaster_testing_data_info.csv")
            img_infos = pd.read_csv(csv_path, index_col='index')

            paths = img_infos["path"].tolist()
            paths = [
                os.path.join(ROOT_DATASET,path)
                for path in paths
            ]

            sigma_test = 50.0
            factor = 16
            list_test_mse = []
            random_state = np.random.RandomState(seed=2204)
            test_i = 0
            s = time.time()
            for file_ in paths:
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()

                img = Image.open(file_)
                img_true_255 = np.array(img).astype(np.float32)
                img_true = img_true_255 / 255.0

                noisy_img_raw = img_true.copy()
                noisy_img_raw += random_state.normal(0, sigma_test/255., img_true.shape)

                noisy_img = torch.from_numpy(noisy_img_raw).permute(2,0,1)
                noisy_img = noisy_img.unsqueeze(0)

                h,w = noisy_img.shape[2], noisy_img.shape[3]
                H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
                padh = H-h if h%factor!=0 else 0
                padw = W-w if w%factor!=0 else 0
                noisy_img = nn.functional.pad(noisy_img, (0,padw,0,padh), 'reflect')

                with torch.no_grad():
                    restored = model(noisy_img.to(DEVICE))

                restored = restored[:,:,:h,:w]
                restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy().copy()

                restored = img_as_ubyte(restored).astype(np.float32)
                test_mse_value = np.square(img_true_255- restored).mean()
                list_test_mse.append(test_mse_value)
                # print(f"test_i={test_i} time={time.time()-s} test_i_psnr_value={20 * np.log10(255.0 / np.sqrt(test_mse_value))}")  
                test_i += 1
                s = time.time()

            psnr_testing = 20 * np.log10(255.0 / np.sqrt(list_test_mse))
            LOGGER.info(f"FINISH TESTING EPOCH {epoch} - iter={i} -  psnr_testing={np.mean(psnr_testing)}")
            model.train()


        i+=1