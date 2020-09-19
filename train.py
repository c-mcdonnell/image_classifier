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

#load a pretrained model

#use GPU if available
device = torch.device("cuda" if torch.cuda.is_available else "cpu")

#load a pretrained model - freeze feature parameters
model = models.vgg16(pretrained = True)
for param in model.parameters():
    param.requires_grad = True

#define feedforward network
classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(25088,4096)),
                            ('relu1', nn.ReLU()),
                            ('dropout1', nn.Dropout(0.2)),
                            ('fc2', nn.Linnear(4096,2048)),
                            ('relu2', nn.ReLU()),
                            ('dropout2', nn.Dropout(0.2)),
                            ('fc3', nn.Linnear(2048, 102)),
                            ('output', nn.LogSoftmax(dim=1))]))
model.classifier = classifier

criterion = nn.NLLLoss()
optimizer = optim.Adam(classifier.parameters(), lr = 0.0001)
model = model.to(device)
images, labels = images.to(device), labels.to(device)