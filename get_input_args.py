#create parser
import argparse

def get_input_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', dest='input_image', action='store', default = 'data_dir'+'/test'+'/4/'+'image_05636.jpg', type = str)
    #parser.add_argument('--save_dir', dest = 'checkpoint.pth', type = str, default='yes', help = 'Would you like to save the checkpoint after training?')
    #parser.add_argument('--arch', type=str, default='vgg16', help = 'CNN model archiecture, default = vgg16')
    #parser.add_argument('--learning_rate', type = int, default=0.0001, help = 'set learning rate for model')
    #parser.add_argument('--hidden_units', type = int, default = 4096, help = 'set number of hidden units in model')
    #parser.add_argument('--epochs', type = int, default = 4, help = 'set number of epochs')
    #parser.add_argument('--gpu', type = str, default = 'yes', help = 'option to use GPU for training')
    #parser.add_argument('--topk', type = int, default = 5, help = 'return the top k most likely classes')
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'map categories to real names')

    in_args = parser.parse_args()

    return in_args