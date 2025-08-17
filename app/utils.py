import cv2
import torch
import numpy as np
import random
import csv
import os

import albumentations as A


def get_imgs_names(path_to_img):
    return list(filter(lambda name: (name.split('.')[1] == "jpg" or name.split('.')[1] == "png"), os.listdir(path_to_img)))

def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def convert_lables_to_dict(path):
    lables = {}
    with open(path) as file:
        reader = csv.reader(file)
        next(reader)
        
        for row in reader:
            lables[row[0]] = int(row[1])
            
    return lables

MODEL_INPUT_SIZE = (288, 288)

val_transformations = A.Compose([
    A.Resize(height=300, width=300, interpolation=cv2.INTER_LINEAR),
    A.CenterCrop(height=MODEL_INPUT_SIZE[0], width=MODEL_INPUT_SIZE[1]),
    A.Normalize(mean=[0.49050629, 0.51356942, 0.46767225], std=[0.18116629, 0.18073399, 0.19205182], always_apply=True),
])

train_transformations = A.Compose([
    A.Resize(height=300, width=300, interpolation=cv2.INTER_LINEAR),
    A.CenterCrop(height=MODEL_INPUT_SIZE[0], width=MODEL_INPUT_SIZE[1]),
    A.OneOf([
        A.CLAHE(clip_limit=4, tile_grid_size=(8, 8), p=0.5),
        A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.6),
        A.GaussNoise(var_limit=(10, 50), mean=0, per_channel=True, p=0.5),
        A.GaussianBlur(p=0.4),
        A.RandomToneCurve(p=0.5)
    ]),
    A.OneOf([
        A.HorizontalFlip(p=0.5),
        # A.RandomRotate90(p=0.4),
        A.Rotate(limit=45, p=0.5),
    ]),
    A.Normalize(mean=[0.49050629, 0.51356942, 0.46767225], std=[0.18116629, 0.18073399, 0.19205182], always_apply=True),
])