# PROGRAMMER: Shih Chieh Chen
# DATE CREATED:2020-03-11
# REVISED DATE:2020-03-11

'''
import nessary functions
'''
import argparse
import all_f

import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict
import time



'''
Define command line Arguments
'''
# Create Parse using ArgumentParser
parser = argparse.ArgumentParser()
# Create command line arguments as mentioned in the introduction part2
parser.add_argument('data_dir', action='store', default='./flowers')
parser.add_argument('-s', '--save_dir', dest='save_dir', action='store', default = './checkpoint.pth')
parser.add_argument('--arch', dest='arch', type = str, action = 'store', default = 'vgg13', help ='Model of CNN vgg13 or vgg16')
parser.add_argument('-lr', '--learning_rate', dest='learning_rate', action = 'store', default = 0.001, help='learning rate value')
parser.add_argument('--hidden_units', dest ='hidden_units', action = 'store', default = 512, type = int, help='hidden_units')
parser.add_argument('-e', '--epochs', dest = 'epochs', action = 'store', default = 20, type = int, help='epochs')
parser.add_argument('--gpu', dest='gpu', action ='store', default='gpu')

# set vars
args = parser.parse_args()
location = args.data_dir
path = args.save_dir
structure = args.arch
lr = args.learning_rate
hidden_layer = args.hidden_units
epochs = args.epochs
gpu = args.gpu

def main():
    # load and process image
    print('Loading data....')
    trainloader, validloader, testloader, train_datasets = all_f.load_data(location)
    print('finish')
    print('setting NN model')
    # set up nn_Network (net_arch, hidden_layer, lr, gpu)
    model, optimizer, criterion = all_f.net_setup(structure, hidden_layer, lr, gpu)
    print('finish')
    print('Start Training...')
    # train the network
    all_f.training_network(model, criterion, optimizer, trainloader, validloader, epochs)
    print('finish training...lol, now save to check point')
    # save checkpoint
    all_f.save_checkpoint(model, optimizer, train_datasets, epochs, path, lr, hidden_layer, structure)
    print('Your checkpoint saved')


if __name__ == '__main__':
    main()
