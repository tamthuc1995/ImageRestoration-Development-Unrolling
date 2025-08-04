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
from torch.optim.lr_scheduler import MultiStepLR

#########################################################################################################
torch.set_float32_matmul_precision('high')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ROOT_PROJECT = "/home/jovyan/shared/Thuc/hoodsgatedrive/projects/ImageRestoration-Development-Unrolling/"
ROOT_DATASET = "/home/jovyan/shared/Thuc/hoodsgatedrive/projects/"

#########################################################################################################

sys.path.append(os.path.join(ROOT_PROJECT, 'exploration/model_multiscale_mixture_GLR/lib'))
from dataloader import ImageSuperResolution
import model_GLR_GTV_deep_v5 as model_structure


LOG_DIR = os.path.join(ROOT_PROJECT, "exploration/model_multiscale_mixture_GLR/result/model_test24/logs/")
LOGGER = logging.getLogger("main")
logging.basicConfig(
    format='%(asctime)s: %(message)s', 
    datefmt='%m/%d/%Y %I:%M:%S %p',
    filename=os.path.join(LOG_DIR, 'training00.log'), 
    level=logging.INFO
)

CHECKPOINT_DIR = os.path.join(ROOT_PROJECT, "exploration/model_multiscale_mixture_GLR/result/model_test24/checkpoints/")
VERBOSE_RATE = 1000

(H_train01, W_train01) = (64, 64)
(H_train02, W_train02) = (128, 128)
(H_train03, W_train03) = (256, 256)
(H_train04, W_train04) = (512, 512)

(H_val, W_val) = (128, 128)
(H_test, W_test) = (496, 496)

train_dataset01 = ImageSuperResolution(
    csv_path=os.path.join(ROOT_DATASET, "dataset/DFWB_training_data_info.csv"),
    dist_mode="vary_addictive_noise",
    lambda_noise=[[1.0, 10.0, 15.0, 20.0, 25.0], [0.1, 0.1, 0.1, 0.1, 0.6]],
    use_data_aug=True,
    patch_size=(H_train01,H_train01),
    patch_overlap_size=(H_train01//4,H_train01//4),
    max_num_patchs=3200000,
    root_folder=ROOT_DATASET,
    logger=LOGGER,
    device=torch.device("cpu"),
)
data_train_batched01 = torch.utils.data.DataLoader(
    train_dataset01, batch_size=16, num_workers=4
)

train_dataset02 = ImageSuperResolution(
    csv_path=os.path.join(ROOT_DATASET, "dataset/DFWB_training_data_info.csv"),
    dist_mode="vary_addictive_noise",
    lambda_noise=[[1.0, 10.0, 15.0, 20.0, 25.0], [0.1, 0.1, 0.1, 0.1, 0.6]],
    use_data_aug=True,
    patch_size=(H_train02,H_train02),
    patch_overlap_size=(H_train02//2,H_train02//2),
    max_num_patchs=1200000,
    root_folder=ROOT_DATASET,
    logger=LOGGER,
    device=torch.device("cpu"),
)
data_train_batched02 = torch.utils.data.DataLoader(
    train_dataset02, batch_size=4, num_workers=4
)

train_dataset03 = ImageSuperResolution(
    csv_path=os.path.join(ROOT_DATASET, "dataset/DFWB_training_data_info.csv"),
    dist_mode="vary_addictive_noise",
    lambda_noise=[[1.0, 10.0, 15.0, 20.0, 25.0], [0.1, 0.1, 0.1, 0.1, 0.6]],
    use_data_aug=True,
    patch_size=(H_train03,H_train03),
    patch_overlap_size=(H_train03//2,H_train03//2),
    max_num_patchs=300000,
    root_folder=ROOT_DATASET,
    logger=LOGGER,
    device=torch.device("cpu"),
)
data_train_batched03 = torch.utils.data.DataLoader(
    train_dataset03, batch_size=2, num_workers=4
)

train_dataset04 = ImageSuperResolution(
    csv_path=os.path.join(ROOT_DATASET, "dataset/DFWB_training_data_info.csv"),
    dist_mode="vary_addictive_noise",
    lambda_noise=[[1.0, 10.0, 15.0, 20.0, 25.0], [0.1, 0.1, 0.1, 0.1, 0.6]],
    use_data_aug=True,
    patch_size=(H_train04,H_train04),
    patch_overlap_size=(H_train04//2,H_train04//2),
    max_num_patchs=200000,
    root_folder=ROOT_DATASET,
    logger=LOGGER,
    device=torch.device("cpu"),
)

data_train_batched04 = torch.utils.data.DataLoader(
    train_dataset04, batch_size=2, num_workers=4
)

validation_dataset = ImageSuperResolution(
    csv_path=os.path.join(ROOT_DATASET, "dataset/CBSD68_testing_data_info.csv"),
    dist_mode="addictive_noise",
    lambda_noise=25.0,
    patch_size=(H_val,H_val),
    patch_overlap_size=(H_val//2,H_val//2),
    max_num_patchs=1000000,
    root_folder=ROOT_DATASET,
    logger=LOGGER,
    device=torch.device("cpu"),
)

test_dataset = ImageSuperResolution(
    csv_path=os.path.join(ROOT_DATASET, "dataset/McMaster_testing_data_info.csv"),
    dist_mode="addictive_noise",
    lambda_noise=25.0,
    patch_size=(H_test,H_test),
    patch_overlap_size=(0,0),
    max_num_patchs=1000000,
    root_folder=ROOT_DATASET,
    logger=LOGGER,
    device=torch.device("cpu"),
)

data_valid_batched = torch.utils.data.DataLoader(
    validation_dataset, batch_size=16, num_workers=4
)

data_test_batched = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, num_workers=4
)

NUM_EPOCHS = 1


model = model_structure.MultiScaleSequenceDenoiser(device=DEVICE)

s = 0
for p in model.parameters():
    s += np.prod(np.array(p.shape))
    # print(p.dtype, np.array(p.shape), s)

LOGGER.info(f"Init model with total parameters: {s}")

criterian = nn.L1Loss()
optimizer = Adam(
    model.parameters(),
    lr=0.0004,
    eps=1e-08
)
lr_scheduler = MultiStepLR(
    optimizer,
    milestones=[200000, 500000, 650000], gamma=0.5
)

### TRAINING
LOGGER.info("######################################################################################")
LOGGER.info("BEGIN TRAINING PROCESS")
# training_state_path = os.path.join(CHECKPOINT_DIR, 'checkpoints_epoch01_iter0300k.pt')
# training_state = torch.load(training_state_path)
# model.load_state_dict(training_state["model"])
# optimizer.load_state_dict(training_state["optimizer"])
# lr_scheduler.load_state_dict(training_state["lr_scheduler"])
# i_checkpoint=training_state["i"]


for epoch in range(NUM_EPOCHS):

    model.train()

    i = 0
    ### TRAINING
    list_train_mse = []
    list_train_psnr = []
    combined_dataloader = itertools.chain(data_train_batched01, data_train_batched02, data_train_batched03, data_train_batched04, data_train_batched04, data_train_batched04)
    for patchs_noisy, patchs_true in combined_dataloader:
        s = time.time()
        optimizer.zero_grad()
        patchs_noisy = patchs_noisy.to(DEVICE)
        patchs_true = patchs_true.to(DEVICE) 
        reconstruct_patchs = model(patchs_noisy.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        loss_value = criterian(reconstruct_patchs, patchs_true)
        loss_value.backward()
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

        if (i%(5*VERBOSE_RATE) == 0):
            checkpoint = { 
                'i': i,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, f'checkpoints_epoch{str(epoch).zfill(2)}_iter{str(i//VERBOSE_RATE).zfill(4)}k.pt'))


        if (i%(VERBOSE_RATE//2) == 0):
            # LOGGER.info(f"Start VALIDATION EPOCH {epoch} - iter={i}")

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
                    # print(f"val_i={val_i} time={time.time()-s} val_i_psnr_value={10 * np.log10(1/val_mse_value)}")
                val_i+=1

            psnr_validation = 10 * np.log10(1/np.array(list_val_mse))
            LOGGER.info(f"FINISH VALIDATION EPOCH {epoch} - iter={i} -  psnr_validation={np.mean(psnr_validation)}")
            model.train()

        if (i%VERBOSE_RATE == 0):

            csv_path = os.path.join(ROOT_DATASET, "dataset/McMaster_testing_data_info.csv")
            img_infos = pd.read_csv(csv_path, index_col='index')

            paths = img_infos["path"].tolist()
            paths = [
                os.path.join(ROOT_DATASET,path)
                for path in paths
            ]

            sigma_test = 25.0
            factor = 8
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
            LOGGER.info(f"FINISH TESING EPOCH {epoch} - iter={i} -  psnr_testing={np.mean(psnr_testing)}")
            model.train()


            # LOGGER.info(f"Start VALIDATION EPOCH {epoch} - iter={i}")
            # ### VALIDAING
            # model.eval()
            # list_test_mse = []
            # test_i = 0
            # for test_patchs_noisy, test_patchs_true in data_test_batched:
            #     s = time.time()
            #     with torch.no_grad():
            #         test_patchs_noisy = test_patchs_noisy.to(DEVICE)
            #         test_patchs_true = test_patchs_true.to(DEVICE) 
            #         reconstruct_patchs = model(test_patchs_noisy.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            #         img_true = np.clip(test_patchs_true[0].cpu().numpy(), a_min=0.0, a_max=1.0).astype(np.float64)
            #         img_recon = np.clip(reconstruct_patchs[0].cpu().numpy(), a_min=0.0, a_max=1.0).astype(np.float64)
            #         test_mse_value = np.square(img_true- img_recon).mean()
            #         list_test_mse.append(test_mse_value)
            #         # LOGGER.info(f"test_i={test_i} time={time.time()-s} test_i_psnr_value={10 * np.log10(1/test_mse_value)}")
            #     test_i+=1

            # psnr_testing = 10 * np.log10(1/np.array(list_test_mse))
            # LOGGER.info(f"FINISH TESING EPOCH {epoch} - iter={i} -  psnr_testing={np.mean(psnr_testing)}")
            # model.train()

        i+=1