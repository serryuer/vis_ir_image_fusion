import os

import torch
from torch.utils.data import Dataset
from PIL import  Image
from torchvision import transforms
import os, sys
import cv2
import numpy as np

sys.path.append('/data/yujsh/xiaoxiannv/fusion')

class TNODataset(Dataset):
    def __init__(self, root, device_id=-1, ):
        self.imgs_r = [os.path.join(os.path.join(root, 'IR'), file) for file in os.listdir(os.path.join(root, 'IR'))]
        self.imgs_v = [os.path.join(os.path.join(root, 'IV'), file) for file in os.listdir(os.path.join(root, 'IV'))]
        self.loader = transforms.Compose([transforms.ToTensor()])
        self.device_id = device_id

    def __getitem__(self, item):
        img_v = cv2.imread(self.imgs_v[item],-1).astype(np.float32)
        img_r = cv2.imread(self.imgs_r[item],-1).astype(np.float32)
        img_r = (img_r/16382*255).astype(np.uint8)
        img_v = (img_v/1393*255).astype(np.uint8)
        img_v = self.loader(img_v)
        img_r = self.loader(img_r)
        return torch.stack([img_v, img_r]).cuda(self.device_id) if self.device_id != -1 else torch.stack([img_v, img_r])

    def __len__(self):
        return len(self.imgs_r)

if __name__ == '__main__':
    dataset = TNODataset('/data/yujsh/xiaoxiannv/fusion/dataset16/train_dataset')
    print(f'dataset length : {len(dataset)}')
    for data in dataset:
        if data.shape[1] == 3:
            print(data.shape)
