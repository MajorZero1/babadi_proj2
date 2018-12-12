import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from scipy import misc
import csv
import os

class MonkeyDataset(Dataset):
    def __init__(self, path_to_data,transform=None):
         self.transform=transform
         self.data_file_list = []
         self.labels = []
         for i in range(0,10):
              path = path_to_data + ("/n%d/" % i)
              files = os.listdir( path )
              for f in files:
                  if f.endswith(".jpg"):
                      self.data_file_list.append(path + f)
                      self.labels.append(i)
                      
    def __len__(self):
         return len(self.labels)
         
    #torchvision.transforms.Resize(size, interpolation=2)
    #torchvision.transforms.ToPILImage(mode=None)
    def __getitem__(self,idx):
        image=misc.imread(self.data_file_list[idx])
        if self.transform:
            image = self.transform(image)
        label=torch.tensor(self.labels[idx])
        return (image,label)
        