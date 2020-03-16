# PROGRAMMER: Shih Chieh Chen
# DATE CREATED:2020-03-15
# REVISED DATE:2020-03-15

### import nessary functions
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse
import time


def load_data(location = './flowers'):
    '''
    load data and convert it for NN network used
    '''

    data_dir = location
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # defint transform for training data
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # define transform for validation data
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # define transform for test data
    test_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    #Load train data with ImageFolder
    train_datasets = datasets.ImageFolder(data_dir + '/train', transform = train_transforms)
    #Load valid data
    valid_datasets = datasets.ImageFolder(data_dir + '/train', transform = valid_transforms)
    #Load test data
    test_datasets = datasets.ImageFolder(data_dir + '/test', transform = test_transforms)

    # define trainloader
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size = 64, shuffle = True)
    # define validloader
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size = 64)
    # define testloader
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size = 64)

    return trainloader, validloader, testloader, train_datasets



### network set up
arch = {'vgg13': 25088, 'vgg16': 25088}
#define nn network
def net_setup(structure='vgg13', hidden_layer = 512, lr = 0.001, gpu = 'gpu'):
    # Load a pre-train network net_arch default is 'vgg13'
    if structure == 'vgg13':
        model = models.vgg13(pretrained = True)
    elif structure == 'vgg16':
        model = models.vgg16(pretrained = True)
    else:
        print('The model {} is not available. The model vgg13 and vgg available now'.format(structure))


    # Freeze parameters
    for parameters in model.parameters():
        parameters.requires_grad = False

        from collections import OrderedDict

        classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(arch[structure], 1024)),
                                                ('relu1', nn.ReLU(inplace = True)),
                                                ('drop1', nn.Dropout(0.4)),
                                                ('fc2', nn.Linear(1024, hidden_layer)),
                                                ('relu2', nn.ReLU(inplace = True)),
                                                ('drop2', nn.Dropout(0.4)),
                                                ('fc3', nn.Linear(hidden_layer, 102)),
                                                ('output', nn.LogSoftmax(dim = 1))]))
        model.classifier = classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr)
    # move to gpu
        if torch.cuda.is_available() and gpu == 'gpu':
            model.cuda()

        return model, optimizer, criterion

def training_network(model, criterion, optimizer, trainloader, validloader, epochs):
    train_losses, valid_losses = [], []
    start = time.time()
    print('The network start training.. please wait.')
    for epoch in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            images, labels = images.to('cuda'), labels.to('cuda') # move images, labels tensor to the default device

        #set optimizer to zero
            optimizer.zero_grad()

        #forward pass
            logps = model.forward(images)
            loss = criterion(logps, labels)

        #bacward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        else:
            valid_loss = 0
            accuracy = 0
            model.eval()

        #trun off graidents
            with torch.no_grad():
                for images, labels in validloader:
                    images, labels = images.to('cuda'), labels.to('cuda')

                    logps = model.forward(images)
                    batch_loss = criterion(logps, labels)
                    valid_loss += batch_loss.item()

                    # calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim = 1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            #back to training mode
            model.train()
            train_losses.append(running_loss/len(trainloader))
            valid_losses.append(valid_loss/len(validloader))

            #print the observation

        print("Epoch: {} / {}..".format(epoch + 1, epochs),
                  "Train loss: {:.3f}..".format(train_losses[-1]),
                  "Valid loss: {:.3f}..".format(valid_losses[-1]),
                  "Valid accuracy: {:.3f}".format(accuracy / len(validloader)))
        print(f"Time per batch: {(time.time() - start)/10:.3f} seconds")
    print('Finish training')
    print('Epochs:{}'.format(epochs))


def save_checkpoint(model, optimizer, train_datasets, epochs, path, lr, hidden_layer, structure):

    model.class_to_idx = train_datasets.class_to_idx
    model.cpu
    #model.class_to_idx = class_to_idx
    torch.save({'arch': structure,
                'hidden_layer': hidden_layer,
                'epoch': epochs,
                'learning_rate': lr,
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx}, path)

#load
def load_checkpoint(path_checkpoint):
    checkpoint = torch.load(path_checkpoint)
    ##指定變數給我們之前設計的模型檔案 model = network_set.net_setup()
   
    structure = checkpoint['arch']
    hidden_layer = checkpoint['hidden_layer']
    lr = checkpoint['learning_rate']

    model,_,_ = net_setup(structure, hidden_layer, lr)
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

# process image
def process_image(path_img):

    pil_im = Image.open(path_img)

    # here process image different with part1
    process_image_into_t = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    t_image = process_image_into_t(pil_im)

    return t_image


def predict(path_img, model, topk_k=5, gpu = 'gpu'):
    if torch.cuda.is_available() and gpu == 'gpu':
        model.to('cuda:0')

    image_pred = process_image(path_img)
    image_pred = image_pred.unsqueeze_(0).float()

    if gpu == 'gpu':
        with torch.no_grad():
            logp = model.forward(image_pred.cuda())
            # calculate the probabilities
    ps = F.softmax(logp.data, dim = 1)

    return ps.topk(topk_k)
