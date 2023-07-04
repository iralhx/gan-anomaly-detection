import os
import cv2
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

class AnomalySet(data.Dataset):
    def __init__(self,data_folder,image_size=256,shuffle = False):
        self.data_folder = data_folder
        ok_img_path=f'{data_folder}/OK'
        ng_img_path=f'{data_folder}/NG'
        self.h=image_size
        self.w=image_size
        self.filenames=[]
        imgs = os.listdir(ok_img_path)
        if shuffle:
            random.shuffle(imgs)
        for img in imgs:
            self.filenames .append(f'{ok_img_path}/{img}')
    def __getitem__(self, index):
        image = Image.open(self.filenames[index])
        data = self.proprecess(image)
        return data

    def __len__(self):
        return len(self.filenames)

    def proprecess(self,data):
        transform_train_list = [
            transforms.Resize((self.h, self.w), interpolation=1),
            transforms.ToTensor()
            ]
        return transforms.Compose(transform_train_list)(data)
    
