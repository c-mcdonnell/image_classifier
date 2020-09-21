#import necessary packages
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import numpy as np

from get_input_args import get_input_args

#Retrieve command line arguments from user running the program from a terminal window
#Returns the collection of these CL arguments from the function call 
in_args = get_imput_args()

#check command line arguments
check_command_line_arguments(in_args)

#load the checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained = True)

    #load saved attributes to model
    model.batch_size = checkpoint['batch_size']
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

def predict(image_path, model, topk = 5):
     ''' Predict the class (or classes) of an image using a trained deep learning model.
     '''
     #convert image file to a processed tensor
     torch_image = torch.from_numpy(process_image(image_path)).type(torch.FloatTensor)
     torch_image = torch_image.unsqueeze(0)
     model.to('cpu')
     model.eval()
     torch_image = torch_image.to('cpu')

     with torch.no_grad():
         #run model on image
         output = model(torch_image)

         #convert model output
         probabilities = torch.exp(output)
         top_probs, top_classes = probabilities.topk(5)

         return top_probs, top_classes