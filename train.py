# import necessary packages

%matplotlib inline 
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import helper
from PIL import Image
import numpy as np

#sample data directory
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#define transforms for training, validation, and testing sets

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomResizedCrop(224),
                                       transforms.ToTensor(),  
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.ToTensor(),  
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

testing_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.ToTensor(),  
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

#Load the datasets with ImageFolder
train_data = dtasets.ImageFolder(train_dir, transform = train_transforms)
valid_data = datasets.ImageFolder(train_dir, transform = valid_transforms)
testing_data = datasets.ImageFolder(train_dir, transform = testing_transforms)

#define dataloaders
trainloader = torch.utils.data.Dataloader(train_data, batch_size = 32, shuffle = True)
validloader = torch.utils.data.Dataloader(valid_data, batch_size = 32, shuffle = True)
testloader = torch.utils.data.Dataloader(testing_data, batch_size = 32, shuffle = True)

#loop through dataloaders to get one batch
for images, labels in trainloader:
    pass

images, labels = next(iter(trainloader))

#load in label mapping
import json
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)