
import os 
from PIL import Image
import numpy as np
import pandas as pd
import skimage

import torch
from torch.utils.data import Dataset

from environ.utils.custom_logger import get_root_logger



class AddictiveGaussianNoiseImagePair(Dataset):
    def __init__(self, 
        csv_path,
        dist_mode="",
        lambda_noise=None,
        patch_size=64,
        patch_overlap_size=32,
        max_num_patchs=100000,
        root_folder="/home/dotamthuc/Works/Projects/unrollGTV/data",
        logger_name=None,
        device_str=None,
    ):

        self.img_infos = pd.read_csv(csv_path, index_col='index')
        self.patch_size = patch_size
        self.patch_overlap_size = patch_overlap_size
        self.max_num_patchs = max_num_patchs
        self.device = torch.device(device_str)
        self.root_folder = root_folder
        self.lambda_noise = lambda_noise
        self.dist_mode = dist_mode
        self.logger = get_root_logger(logger_name)
        self.patchs_data_all = pd.DataFrame({})
        self.create_patchs()
        self.max_num_patchs = min(max_num_patchs, self.patchs_data_all.shape[0])
        self.random_state = None
        self.random_permute(seed=2204)


    def __len__(self):
        return len(self.patchs_data)
    
    def RGB2YCbCr(self, rgb):
        return skimage.color.rgb2ycbcr(rgb)

    def YCbCr2RGB(self, ycbcr):
        return skimage.color.ycbcr2rgb(ycbcr)
    
    def create_patchs(self):
        list_pdf_per_img = []
        for i in range(self.img_infos.shape[0]):
            img_info = self.img_infos.loc[i]
            height = img_info["height"]
            width  = img_info["width"]
            
            width_jumps = np.arange(0, width-self.patch_size, self.patch_size - self.patch_overlap_size)
            height_jumps = np.arange(0, height-self.patch_size, self.patch_size - self.patch_overlap_size)
            xindex, yindex = np.meshgrid(width_jumps, height_jumps)
            xy_location = np.stack([yindex, xindex], axis=2).reshape(-1, 2)
            pdf_patchs = pd.DataFrame({
                "row": xy_location[:, 0],
                "col": xy_location[:, 1],
            })

            pdf_patchs["path"] = os.path.join(
                self.root_folder,
                img_info["path"]
            )
            list_pdf_per_img.append(pdf_patchs)

        self.patchs_data_all = pd.concat(list_pdf_per_img)
        self.logger.info(f"Dataset - Create total {self.patchs_data_all.shape[0]} patchs")

    def random_permute(self, seed=2204):
        self.logger.info(f"Dataset - Permute and select {self.max_num_patchs}/{self.patchs_data_all.shape[0]} patchs")
        self.random_state = np.random.RandomState(seed=seed)
        ind = self.random_state.permutation(self.max_num_patchs)
        self.patchs_data = self.patchs_data_all.iloc[ind].copy()

    def __getitem__(self, idx):
        row, col, path = tuple(self.patchs_data.iloc[idx])
        self.logger.info(f"Loadding patch: row={row}, col={col}, path={path}")

        img = Image.open(path)
        img = np.array(img)
        
        patch = img[row:row + self.patch_size, col: col + self.patch_size, :]

        h = patch.shape[0]
        w = patch.shape[1]
        h_ = (h//16) * 16
        w_ = (w//16) * 16
        patch = patch[0:h_, 0:w_]

        # patch = self.RGB2YCbCr(patch).astype(np.float32) / 255.0
        patch = patch.astype(np.float32) / 255.0
        if (self.dist_mode == "addictive_noise"):
            noise = self.random_state.normal(loc=0.0, scale=self.lambda_noise / 255.0, size=(h_, w_, 3))
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


