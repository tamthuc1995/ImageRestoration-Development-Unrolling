
import os 

from PIL import Image
import numpy as np

import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pyplot import imread

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.nn.parameter import Parameter

import skimage
import logging

# LOGGER = logging.getLogger("main")

def data_augmentation(image, mode):
    """
    Performs data augmentation of the input image
    Input:
        image: a numpy array image
        mode: int. Choice of transformation to apply to the image
                0 - no transformation
                1 - flip up and down
                2 - rotate counterwise 90 degree
                3 - rotate 90 degree and flip up and down
                4 - rotate 180 degree
                5 - rotate 180 degree and flip
                6 - rotate 270 degree
                7 - rotate 270 degree and flip
    Using np.random.randint(0,7) for mode randomness
    """
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    return out.copy()



class ImageSuperResolution(Dataset):
    def __init__(self, 
        csv_path,
        dist_mode="",
        lambda_noise=None,
        use_data_aug=False,
        patch_size=(64,64),
        max_num_patchs=100000,
        root_folder="/home/dotamthuc/Works/Projects/unrollGTV/data",
        logger=None,
        device=None,
    ):

        self.img_infos = pd.read_csv(csv_path, index_col='index')
        self.patch_size = patch_size
        self.max_num_patchs = max_num_patchs
        self.device = device
        self.root_folder = root_folder
        self.lambda_noise = lambda_noise
        self.use_data_augmentation = use_data_aug
        self.dist_mode = dist_mode
        self.logger = logger
        self.patchs_data_all = pd.DataFrame({})
        self.create_all_images()

        self.random_state = np.random.RandomState(seed=2204)
        self.max_num_patchs = max_num_patchs
        self.create_patchs(self.max_num_patchs)
        self.random_permute_subselect_patchs_data(self.max_num_patchs)


    def __len__(self):
        return len(self.patchs_data)
    
    def RGB2YCbCr(self, rgb):
        return skimage.color.rgb2ycbcr(rgb)

    def YCbCr2RGB(self, ycbcr):
        return skimage.color.ycbcr2rgb(ycbcr)
    
    def create_all_images(self):
        img_size = 512
        overlap = 96
        max_size = 800
        list_pdf_per_img = []
        for i in range(self.img_infos.shape[0]):
            img_info = self.img_infos.loc[i]
            height = img_info["height"]
            width  = img_info["width"]
            nchannels  = img_info["nchannels"]
            if (width > max_size) and (height > max_size):
                height_jumps = np.arange(0, height-img_size, img_size - overlap)
                width_jumps = np.arange(0, width-img_size, img_size - overlap)
                xindex, yindex = np.meshgrid(width_jumps, height_jumps)
                xy_location = np.stack([yindex, xindex], axis=2).reshape(-1, 2)
                pdf_patchs = pd.DataFrame({
                    "row": xy_location[:, 0],
                    "col": xy_location[:, 1],
                })
                pdf_patchs["height"] = img_size
                pdf_patchs["width"] = img_size
                pdf_patchs["nchannels"] = nchannels
                pdf_patchs["path"] = os.path.join(
                    self.root_folder,
                    img_info["path"]
                )
                list_pdf_per_img.append(pdf_patchs)
            else:
                pdf_patchs = pd.DataFrame({
                    "row": [0],
                    "col": [0],
                })
                pdf_patchs["height"] = height
                pdf_patchs["width"] = width
                pdf_patchs["nchannels"] = nchannels
                pdf_patchs["path"] = os.path.join(
                    self.root_folder,
                    img_info["path"]
                )
                list_pdf_per_img.append(pdf_patchs)

        self.images_data_all = pd.concat(list_pdf_per_img).reset_index(drop=True)
        self.logger.info(f"Dataset - Create total {self.images_data_all.shape[0]} cropped images")

    def create_patchs(self, max_num_patchs):
        list_pdf_per_img = []
        N_loops = (max_num_patchs // self.images_data_all.shape[0] + 1)
        for loop in range(N_loops):
            for i in range(self.images_data_all.shape[0]):
                img_info = self.images_data_all.iloc[i]
                # print(img_info)
                height = img_info["height"]
                width  = img_info["width"]
                row = img_info["row"]
                col  = img_info["col"]
                nchannels  = img_info["nchannels"]
                if nchannels > 3:
                    continue

                if (self.patch_size[0] < height) and (self.patch_size[1] < width):
                    patchs = {
                        "row": row + self.random_state.randint(0, height - self.patch_size[0]),
                        "col": col + self.random_state.randint(0, width - self.patch_size[1]),
                        "path": img_info["path"]
                    }
                    list_pdf_per_img.append(patchs)
                else:
                    assert False, "patch_size too big"

        self.patchs_data_all = pd.DataFrame(list_pdf_per_img)
        self.logger.info(f"Dataset - Create total {self.patchs_data_all.shape[0]} patchs")

    def random_permute_subselect_patchs_data(self, max_num_patchs):
        self.logger.info(f"Dataset - Permute and select {max_num_patchs}/{self.patchs_data_all.shape[0]} patchs")
        ind = self.random_state.permutation(self.patchs_data_all.shape[0])[:max_num_patchs]
        self.patchs_data = self.patchs_data_all.iloc[ind].copy()

    def __getitem__(self, idx):
        row, col, path = tuple(self.patchs_data.iloc[idx])
        img = Image.open(path)
        img = np.array(img)

        patch = img[row:row + self.patch_size[0], col: col + self.patch_size[1], :]

        h = patch.shape[0]
        w = patch.shape[1]
        h_ = (h//8) * 8
        w_ = (w//8) * 8
        patch = patch[0:h_, 0:w_]

        # patch = self.RGB2YCbCr(patch).astype(np.float32) / 255.0
        if self.use_data_augmentation:
            arg_mode = self.random_state.randint(0, 7)
            patch = data_augmentation(patch, arg_mode)

        patch = patch.astype(np.float32) / 255.0
        if (self.dist_mode == "addictive_noise"):
            noise = self.random_state.normal(loc=0.0, scale=self.lambda_noise / 255.0, size=(h_, w_, 3))
            patch_dist = patch + noise.astype(np.float32)

        if (self.dist_mode == "vary_addictive_noise"):
            lambda_noise_selected = self.random_state.choice(self.lambda_noise[0], p=self.lambda_noise[1])
            # print(f"lambda_noise_selected={lambda_noise_selected}")
            noise = self.random_state.normal(loc=0.0, scale=lambda_noise_selected / 255.0, size=(h_, w_, 3))
            patch_dist = patch + noise.astype(np.float32)


        if (self.dist_mode == "addictive_noise_scale"):
            noise = self.random_state.normal(loc=0.0, scale=1.0, size=(h_, w_, 3))
            noise = noise * (self.lambda_noise / 255.0)
            patch_dist = patch + noise.astype(np.float32)


        sample = (
            torch.from_numpy(patch_dist).to(self.device),
            torch.from_numpy(patch).to(self.device)
        )
        
        return sample


