
import numpy as np
import os
import pandas as pd
import torch
import albumentations as A
import albumentations.pytorch
import cv2
import math
import pydicom

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# IMAGE_SIZE = 2048
IMAGE_SIZE= 512

train_transforms = A.Compose([
#     A.Resize(IMAGE_SIZE, IMAGE_SIZE, p=1),
    A.OneOf([
        A.HorizontalFlip(p=.8),
        A.VerticalFlip(p=.8),
        A.RandomRotate90(p=.8)]
    ),
    A.RandomResizedCrop(height=IMAGE_SIZE,width=IMAGE_SIZE,scale=[0.8,1.0],ratio=[0.8,1.2],p=1.),
    A.OneOf([
        A.transforms.JpegCompression(quality_lower=99, quality_upper=100, p=.7),
#         A.transforms.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.2, alpha_coef=0.08, p=.7),
        A.transforms.RandomGamma(gamma_limit=(80, 120), eps=None, p=.7),
    ]),
#     A.transforms.ColorJitter(brightness=0.05, contrast=0.2, saturation=0.2, hue=0.02, p=.7),
    A.pytorch.ToTensor(), 
], p=1.0)    

val_transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE, p=1),
    A.pytorch.ToTensor(),     
], p=1.0)    

test_transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE, p=1),
    A.pytorch.ToTensor(),     
], p=1.0) 


class RsnaDataset(Dataset):
    def __init__(self, df, transforms=None, path='../data/stage_2_train_images/'):
        self.df = df
        self.transforms = transforms if transforms else None
        self.dir = path

    def __len__(self):
        return len(self.df)
    
    def read_dicom_image(self, loc):
        # return numpy array
        img_arr = pydicom.read_file(loc).pixel_array
        img_arr = img_arr/img_arr.max()
        img_arr = (255*img_arr).clip(0, 255).astype(np.uint8)
        img_arr = Image.fromarray(img_arr).convert('RGB') # model expects 3 channel image
        return img_arr    

    def __getitem__(self, idx):
        pid = self.df.iloc[idx, 0]
#         print(pid)
        filepath = [] 
        filepath.append(self.dir) 
        filepath.append(pid) 
        filepath.append('.dcm')
        filepath = ''.join(filepath)
        pimage = self.read_dicom_image(filepath)
        image = np.array(pimage)
#         print(type(image))
#         print(image.shape)
        if self.transforms:
            timage = self.transforms(image=image)
            image = timage["image"]
        label = self.df.iloc[idx, 5]
        return image, label

def get_rsna_data(args) :    
    
    PATH = 'data/'
    df = pd.read_csv(PATH + 'stage_2_train_labels.csv')
    df = df.drop_duplicates('patientId').reset_index(drop=True)
    
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=0)

    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    print(train_df.shape)
    print(test_df.shape)

    print(len(df[df['Target']==1])/len(df))
    print(len(train_df[train_df['Target']==1])/len(train_df))
    print(len(test_df[test_df['Target']==1])/len(test_df))   
    
    BATCH_SIZE = args.batch_size
    train_dataset = RsnaDataset(train_df, transforms=train_transforms, path='data/stage_2_train_images/')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.workers)

    test_dataset = RsnaDataset(test_df, transforms=test_transforms, path='data/stage_2_train_images/')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.workers)     
    
    return train_loader, test_loader

def get_imagenet_data(args) :
    data_dir = 'ILSVRC/Data/CLS-LOC/'
    traindir = os.path.join(data_dir, 'train')
    valdir = os.path.join(data_dir, 'val') 

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader

