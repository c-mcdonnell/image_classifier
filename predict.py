#import necessary packages
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import numpy as np

#load the checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained = True)

    #load saved attributes to model
    model.batch_size = checkpoint['batch_size']
    model.learning_rate = checkpoint['learing_rate']
    model.state_dict = checkpoint['state_dict']
    model.optimizer = checkpoint['optimizer']
    model.input_size = checkpoint['input_size']
    model.output_size = checkpoint['output_size']
    model.criterion = checkpoint['criterion']
    model.class_to_idx = checkpoint['class_to_idx']

    #freeze parameters to avoid backpropogation
    for param in model.parameters():
        param.requires_grad = False

    return model

saved_model = load_checkpoint('checkpoint.pth')


def process_image(image):
    '''Scales, crops, and normalizes a PIL image for a PyTorch model, 
    returns a numpy array
    '''
    #process a PIL image for use in Pytorch model

    #open image
    im = Image.open(image)

    #transform image
    img_transform = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                            std = [0.229, 0.224, 0.225])])

    transformed_img = img_transform(im)

    #convert to numpy array
    np_image = np.array(transformed_img)

    return np_image