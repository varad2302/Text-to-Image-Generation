import bisect
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import clip
import random
import torch
import pickle
import os

class ImagePaths(Dataset):
    def __init__(self, training_images_list_file, transform):
        if 'oxford' in training_images_list_file:
            dataset = 'oxford'
            self.root = '.\\dataset\\flower'
        if 'cub' in training_images_list_file:
            dataset = 'cub'
            self.root = '.\\dataset\\bird'
        if 'coco' in training_images_list_file:
            dataset = 'coco'
            self.root = '.\\dataset\\coco'

        with open(training_images_list_file, 'rb') as f:
            self.caption = pickle.load(f)

        self._length = len(self.caption)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        self.preprocess = preprocess
        self.preprocessor = transform

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = self.preprocessor(image)
        return image

    def __getitem__(self, i):
        cap = self.caption[i]
        path = cap[0]
        path = os.path.join(self.root, path)
        image = self.preprocess_image(path)
        captions = cap[1]
        index = random.randint(0, len(captions)-1)
        caption = captions[index]
        text = clip.tokenize([caption], truncate=True)
        return image, text.reshape(77)
    
    def __getitem__(self, i):
        cap = self.caption[i]
        relative_path = cap[0]
        full_path = os.path.join(self.root, relative_path)
        image = self.preprocess_image(full_path)
        captions = cap[1]
        index = random.randint(0, len(captions)-1)
        caption = captions[index]
        text = clip.tokenize([caption], truncate=True)
        return image, text.reshape(77)