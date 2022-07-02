# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate
import pdb
import torch
from PIL import Image
import json
import numpy as np
import random
import torchvision.transforms as transforms
import os
import cv2 as cv

identity = lambda x:x

class SetDataset:
    def __init__(self, data_file, batch_size, transform):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
 
        self.cl_list = np.unique(self.meta['image_labels']).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
            self.sub_meta[y].append(x)

        self.sub_dataloader = [] 
        
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)        
        for cl in self.cl_list:
            sub_dataset = SubDataset(
                self.sub_meta[cl], cl, transform=transform)
            self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )
        # pdb.set_trace()

    def __getitem__(self,i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)

class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self,i):
        seed = np.random.randint(2147483647) 
        random.seed(seed) 
        torch.manual_seed(seed)
        
        image_path = os.path.join(self.sub_meta[i])
        img = Image.open(image_path).convert('RGB')
        random.seed(seed)
        torch.manual_seed(seed)
        img = self.transform(img)
        # if self.edges:
        #     cv_img = cv.imread(image_path, 0)
        #     edges = cv.Canny(cv_img, 100, 200)
        #     edges = cv.cvtColor(edges, cv.COLOR_BGR2RGB)
        #     edges = Image.fromarray(edges)
        #     random.seed(seed)
        #     torch.manual_seed(seed)
        #     edges = self.transform_edges(edges)
        # else:
        #     edges = 0

        # from matplotlib import pyplot as plt
        # from einops import rearrange
        # img = rearrange(img, 'c h w -> h w c')[..., :3]
        # edges = rearrange(edges, 'c h w -> h w c')[..., :3]
        # plt.subplot(121),plt.imshow(img,cmap = 'gray')
        # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        # plt.subplot(122),plt.imshow(edges,cmap = 'gray')
        # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        # plt.show()
        # pdb.set_trace()
        
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]
