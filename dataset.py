import numpy as np
import cv2 #https://www.jianshu.com/p/f2e88197e81d
import random

from skimage.io import imread
from skimage import color

import torch
import torch.utils.data
from torchvision import datasets, models, transforms

from multiprocessing import Pool, cpu_count
from tqdm import tqdm

save_list = [3,8,47,54,87,97,130,15,26,40,83,122,148,163]

class Dataset(torch.utils.data.Dataset):

    def __init__(self, args, img_paths, mask_paths,aug=False, preload=False):
        self.args = args
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.aug = aug
        self.preload = preload
        # self.index=select_index ##for test

        if self.preload:
            print("Preloading dataset into memory...")
            self.images, self.labels = self._preload_data()
            print("Preloading complete.")
        else:
            self.images, self.labels = None, None
            
    def _load_single_sample(self, paths):
        img_path, mask_path = paths
        npimage = np.load(img_path).astype("float32")
        npmask = np.load(mask_path).astype("float32")
        
        npimage = npimage.astype("float32")
        
        WT_Label = (npmask == 1) | (npmask == 2) | (npmask == 4)
        TC_Label = (npmask == 1) | (npmask == 4)
        ET_Label = (npmask == 4)
        nplabel = np.stack((WT_Label, TC_Label, ET_Label), axis=0).astype("float32")
        return npimage, nplabel

    def _preload_data(self):
        """Preload all data into memory using multiprocessing with a progress bar."""
        with Pool(cpu_count()*4) as pool:
            # Wrap the pool map with tqdm for progress tracking
            data = list(tqdm(pool.imap(self._load_single_sample, zip(self.img_paths, self.mask_paths)),
                             total=len(self.img_paths), desc="Preloading data"))
        images, labels = zip(*data)
        return list(images), list(labels)

    def __len__(self):
        return len(self.img_paths)##1 for test

    def __getitem__(self, idx):
        # Load preloaded data or from disk
        if self.preload:
            npimage = self.images[idx]
            nplabel = self.labels[idx]
        else:
            npimage = np.load(self.img_paths[idx]).astype("float32")
            npmask = np.load(self.mask_paths[idx]).astype("float32")

            npimage = npimage.astype("float32")

            WT_Label = (npmask == 1) | (npmask == 2) | (npmask == 4)
            TC_Label = (npmask == 1) | (npmask == 4)
            ET_Label = (npmask == 4)
            nplabel = np.stack((WT_Label, TC_Label, ET_Label), axis=0).astype("float32")
            
        npimage = npimage.transpose((2, 0, 1))
            
        return npimage, nplabel

