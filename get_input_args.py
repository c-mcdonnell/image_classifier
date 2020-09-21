#create parser
import argparse

def get_input_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type = str, default='yes', help = 'Would you like to save the checkpoint after training?')
    parser.add_argument('--arch', type=str, default='vgg16', help = 'CNN model archiecture, default = vgg16')
    parser.add_argument('--learning_rate', type = int, default=0.0001, help = 'set learning rate for model')
    parser.add_argument('--hidden_units', type = int, default = 4096, help = 'set number of hidden units in model')
    parser.add_argument('--epochs', type = int, default = 4, help = 'set number of epochs')
    parser.add_argument('--gpu', type = str, default = 'yes', help = 'option to use GPU for training')

    in_args = parser.parse_args()

    return in_args