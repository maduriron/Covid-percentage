import os
import torch
from PIL import Image
import re

from torch.utils.data import Dataset, DataLoader
    

class LeukemiaDataset(Dataset):
    def __init__(self, root, fold_id=0, fold_splitter=None, transforms=None,
                replacer=None, prepath=None):
        """
        params:
        root := directory where data is hold
        fold_id := id number of split for current training process
        fold_splitter := {"fold_id0": {"paths": [<<list of paths>>], "metadata": [<<list of metadata>>]},
                        "fold_id1": {"paths": [<<list of paths>>], "metadata": [<<list of metadata>>]},
                        ...
                        }
        transforms := transforms that should be done for current data
        """
        self.root = root
        self.fold_id = fold_id
        self.fold_splitter = fold_splitter
        self.transforms = transforms
        self.replacer = replacer
        self.prepath = prepath 
                
    def __len__(self):
        return len(self.fold_splitter[self.fold_id]["paths"])
    
    def convert_to_distribution(self, value):
        distribution = [0] * 101
        if value >= 2 and value <= 98:
            distribution[value] = 0.6
            distribution[value - 1] = 0.15 
            distribution[value + 1] = 0.15
            distribution[value - 2] = 0.05
            distribution[value + 2] = 0.05
        elif value == 1:
            distribution[value] = 0.6
            distribution[value - 1] = 0.2
            distribution[value + 1] = 0.15
            distribution[value + 2] = 0.05
        elif value == 99:
            distribution[value] = 0.6
            distribution[value - 1] = 0.15
            distribution[value - 2] = 0.05
            distribution[value + 1] = 0.2
        elif value == 0:
            distribution[value] = 0.8
            distribution[value + 1] = 0.15
            distribution[value + 2] = 0.05
        elif value == 100:
            distribution[value] = 0.8
            distribution[value - 1] = 0.15
            distribution[value - 2] = 0.05
        else:
            print("OMG")
        return distribution

    def __getitem__(self, id):
        path = self.fold_splitter[self.fold_id]["paths"][id]
        path = path.replace(self.prepath, self.replacer)
        metadata = self.fold_splitter[self.fold_id]["metadata"][id]
        metadata = int(metadata)
        path = os.path.join(self.root, path)
        img = Image.open(path).convert("RGB")
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        #distribution = self.convert_to_distribution(metadata)
        
        #return img, torch.tensor(distribution), metadata, path
        return img, metadata, path