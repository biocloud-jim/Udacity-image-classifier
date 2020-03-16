# PROGRAMMER: Shih Chieh Chen
# DATE CREATED:2020-03-12
# REVISED DATE:2020-03-12

'''
import nessary functions
'''
import argparse
import all_f
import numpy as np
import PIL
from PIL import Image
import get_data
import json
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision import models
'''
Define command line Arguments
'''
# Create Parse using ArgumentParser
parser = argparse.ArgumentParser()
# Create command line arguments as mentioned in the introduction part2
parser.add_argument('input', action ='store', type = str, default='./flowers/test/1/image_06743.jpg')
parser.add_argument('checkpoint', action ='store', type = str, default = './checkpoint.pth')
parser.add_argument('-k', '--top_k', action ='store', dest = 'top_k', type = int, default = 5)
parser.add_argument('--category_names', action ='store', dest ='category_names', default = './cat_to_name.json')
parser.add_argument('--gpu', action ='store', dest = 'gpu', default = 'gpu', help = 'activate gpu, default is activate')

#set vars
args = parser.parse_args()
path_img = args.input
path_checkpoint = args.checkpoint
output_numbers = args.top_k
gpu = args.gpu

def main():
    # data transforms
    #skip:trainloader, validloader, testloader = get_data.load_data()
    # load saved checkpoint
    model = all_f.load_checkpoint(path_checkpoint)
    print(model)
    #load category
    with open('cat_to_name.json', 'r') as jf:
        cat_to_names = json.load(jf)

    #call predict core function return probs
    output_predict = all_f.predict(path_img, model, output_numbers, gpu = 'gpu')
    #
    labels = [cat_to_names[str(index+1)] for index in np.array(output_predict[1][0])]
    probability = np.array(output_predict[0][0])

    i = 0
    while i < output_numbers:
        print('{} \'s probability of {}'.format(labels[i], probability[i]))
        i += 1
if __name__ == '__main__':
    main()
