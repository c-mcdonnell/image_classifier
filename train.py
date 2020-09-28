# import necessary packages

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import numpy as np

import argparse

#Retrieve command line arguments from user running the program from a terminal window
#Returns the collection of these CL arguments from the function call 
#in_args = get_imput_args()

#check command line arguments
#check_command_line_arguments(in_args)
parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', dest = 'checkpoint.pth', action = 'store', type = str, default='yes', help = 'Would you like to save the checkpoint after training?')
parser.add_argument('--arch', type=str, default='vgg16', help = 'CNN model archiecture, default = vgg16')
parser.add_argument('--learning_rate', type = int, default=0.0001, help = 'set learning rate for model')
parser.add_argument('--hidden_units', type = int, default = 4096, help = 'set number of hidden units in model')
parser.add_argument('--epochs', type = int, default = 4, help = 'set number of epochs')
parser.add_argument('--gpu', type = str, default = 'yes', help = 'option to use GPU for training')
parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'map categories to real names')
in_args = parser.parse_args()

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
train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
valid_data = datasets.ImageFolder(train_dir, transform = valid_transforms)
testing_data = datasets.ImageFolder(train_dir, transform = testing_transforms)

#define dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size = 32, shuffle = True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size = 32, shuffle = True)
testloader = torch.utils.data.DataLoader(testing_data, batch_size = 32, shuffle = True)

#loop through dataloaders to get one batch
for images, labels in trainloader:
    pass

images, labels = next(iter(trainloader))

#load in label mapping
import json
with open(in_args.category_names, 'r') as f:
    in_args.category_names = json.load(f)

#load a pretrained model

#use GPU if available
device = torch.device("cuda" if torch.cuda.is_available and in_args.gpu == 'yes' else "cpu")

#load a pretrained model - freeze feature parameters
architecture = in_args.arch
model = models.architecture(pretrained = True)
for param in model.parameters():
    param.requires_grad = True

#define feedforward network
classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(25088,in_args.hidden_units)),
                            ('relu1', nn.ReLU()),
                            ('dropout1', nn.Dropout(0.2)),
                            ('fc2', nn.Linear(in_args.hidden_units,2048)),
                            ('relu2', nn.ReLU()),
                            ('dropout2', nn.Dropout(0.2)),
                            ('fc3', nn.Linear(2048, 102)),
                            ('output', nn.LogSoftmax(dim=1))]))
model.classifier = classifier

criterion = nn.NLLLoss()
optimizer = optim.Adam(classifier.parameters(), lr = in_args.learning_rate)
model = model.to(device)
images, labels = images.to(device), labels.to(device)

#test, I don't think I need this
'''
i = 0
for images, labels in trainloader:
    model, images = model.to(device), images.to(device)
    log_ps = model(images)
    i += 1
ps = torch.exp(log_ps)
'''
#train feedforward network

#keep session active so it doesn't disconnect while training
'''
do I need this if not in jupyter notebook?
'''
from workspace_utils import active_session
with active_session():
    #get data, initialize parameters
    images, labels = next(iter(trainloader))

    epochs = in_args.epochs
    steps = 0
    #running_loss = 0
    '''
    running_loss is below (line 140), do I need it up here too?
    '''
    print_every = 40
    accuracy = 0
    valid_loss = 0

    #change model to train mode and move it to the GPU
    model.train()
    model.to(device)

    #train
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:

            steps += 1
            images, labels = images.to(device), labels.to(device)
            '''
            I don't think I need this because I already have it globally in line 87
            '''
            #clear out gradients
            optimizer.zero_grad()

            #forward and backward pass
            log_ps = model(images)
            training_loss = criterion(log_ps, labels)
            training_loss.backward()
            optimizer.step()

            running_loss += training_loss.item()

            #during training, every 40 steps, drop out of the training loop and test accuracy on validation set
            #validate (switch to evaluation mode - turns off dropout so we can make accurate predictions)
            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                #control new weights, don't calculate gradient; can continue backpropogation
                    for images2, labels2 in validloader:
                        model.to(device)
                        images2, labels2 = images2.to(device), labels2.to(device)
                        #forward pass
                        log_ps2 = model(images2)
                        batch_loss = criterion(log_ps2, labels2)
                        valid_loss += batch_loss.item()
                        #get top probability
                        ps2 = torch.exp(log_ps2)
                        top_ps2, top_class = ps2.topk(1,dim=1)
                        ##check to see if the top class is equal to the labels
                        #equality = 1 if it's a match and it = 0 if it's wrong
                        equality = top_class == labels2.view(*top_class.shape)
                        #takes the average of equality (rights and wrongs) and continues to increase throught the loops
                        accuracy += torch.mean(equality.type(torch.FloatTensor))
                training_loss = running_loss/print_every
                valid_loss = valid_loss/len(validloader)
                accuracy = accuracy/len(testloader)

        print('epoch {}/{}'.format(e+1, epochs),
                'training loss {:.3f}'.format(training_loss),
                'validation loss {:.3f}'.format(valid_loss),
                'accuracy {:.3f}'.format(accuracy))

    running_loss = 0
    model.train()
print('woohoo! Training is complete!')

#save the checkpoint

#save model to cpu
device = torch.device('cpu')
model = model.to(device)

#assign class_to_idx as an attritube to model
model.class_to_idx = train_data.class_to_idx

#define checkpoint with parameters to be saved

checkpoint = {'model_arch': 'vgg16',
            'batch_size' : 32,
            'lr': in_args.learning_rate,
            'epoch': in_args.epochs,
            'arch': in_args.arch,
            'hidden_units': in_args.hidden_units,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'input_size': 25088,
            'output_size': 4096,
            'criterion': criterion,
            'class_to_idx': model.class_to_idx,
            'classifier': model.classifier}
torch.save(checkpoint, in_args.save_dir)

