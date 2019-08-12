import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import random
from torchvision import transforms
import torchvision.transforms.functional as TF

class KittiLoader(Dataset):
    def __init__(self, img_dir, gt_dir, d, ratio,transform=None):



        self.img_paths = [img_dir+f for f in d]
        self.gt_paths = [gt_dir+f for f in d]
        #print(self.gt_paths)
        idx = list(range(len(self.gt_paths)))
        self.weights = np.array([a % int(1/ratio) == 0 for a in idx])  # , dtype = np.uint8


    def __len__(self):
        return len(self.img_paths)

    def transform(self, image, mask):
        # Resize
        maskk = np.array(mask)
        m = maskk > 0.5
        maskk[m] = 1
        maskk[1 - m] = 0
        mask = Image.fromarray(maskk)

        delta = random.random()*(256-224)
        new_x = 224+int(delta)
        resize = transforms.Resize(size=(new_x, new_x))

        image = resize(image)
        mask = resize(mask)

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(224, 224))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
        # print(image.shape)
        # print(mask.shape)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Transform to tensor
        maskk = np.array(mask)
        m = maskk>0.5
        maskk[m]=1
        maskk[1-m]=0
        mask = Image.fromarray(maskk)
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        return image, mask

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx])
        mask = Image.open(self.gt_paths[idx])
        x, y = self.transform(image, mask)
        w = torch.Tensor([self.weights[idx]])
        return x, y, w




        ######必须保证三维

