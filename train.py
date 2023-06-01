# Imports here
import os
import json
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
#from workspace_utils import active_session
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import argparse

# Train a new network on a data set with train.py
# • Basic usage: python train.py data_directory
# • Prints out training loss, validation loss, and validation accuracy as the network trains
# • Options: 
#   * Set directory to save checkpoints: python train.py data_dir --save_dir save_directory 
#   * Choose architecture: python train.py data_dir --arch "vgg13" 
#   * Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20 
#   * Use GPU for training: python train.py data_dir --gpu

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='Directory with data files', default='flowers')
    parser.add_argument('--save_dir', help='Directory to save checkpoints', default='checkpoint.pth')
    parser.add_argument('--arch', help='Choose Architecture', default='resnet50')
    parser.add_argument('--learning_rate', help='Learning Rate', type=float, default=0.001)
    parser.add_argument('--hidden_units', help='Number of Hidden Units', type=int, default = 512)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default = 1)
    parser.add_argument('--gpu', help='Use GPU', action='store_true')
    args = parser.parse_args()
    return args

def load_data(data_dir):
    print("Loading Data")
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
   
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    return trainloader, validloader, testloader, train_data
    
def create_classifier(model, hidden_units, dropout, output_size):
    print("Will create classifier")
    features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(features, hidden_units), #2048 input features in resnet50
                               nn.ReLU(),
                               nn.Dropout(p=dropout),
                               nn.Linear(hidden_units, output_size), # 102 categories
                               nn.LogSoftmax(dim=1))
    return model, features

def train_model(model, optimizer, criterion, trainloader, validloader, epochs, gpu):
    print("Will Train the model")
    
    step = 0
    print_every = 5
    running_loss = 0
    train_losses, test_losses = [], []
    device = torch.device("cuda:0" if gpu == True else "cpu")
    model.to(device)
    for epoch in range(epochs):
        for images, labels in trainloader:
            step += 1
            images, labels = images.to(device), labels.to(device)
            # training loop
            optimizer.zero_grad()
            logps = model(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_losses.append(running_loss/len(trainloader))

            if step % print_every == 0:
                #with torch.no_grad():
                model.eval()
                test_loss = 0
                accuracy = 0

                for images, labels in validloader:
                #for images, labels in testloader:
                    images, labels = images.to(device), labels.to(device)
                    logps = model(images)
                    loss = criterion(logps, labels)
                    test_loss += loss.item()

                    #check accuracy
                    ps = torch.exp(logps)
                    top_ps, top_class = ps.topk(1, dim=1)
                    equality = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equality.type(torch.FloatTensor))

                running_loss = 0
                model.train()

                print(f"Epoch {epoch+1}/{epochs}.. "
                        f"Train Loss {train_losses[-1]:.3f}.. ")
    # print training loss 
    #return model so it can be validated
    return model

def validate(model, testloader, gpu):
    print("Will Validate model")
    correct = 0
    total = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        model.eval()
        for images, labels in testloader:
            if gpu==True:
                images, labels = images.to('cuda'), labels.to('cuda')
            result = model(images)
            _, predicted = torch.max(result.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(result, labels)
            total_loss += loss.item()
    print(f'Test Accuracy: {correct / total * 100:.2f}%')
    print(f'Test Loss: {loss:.2f}%')

def save_checkpoint(model, save_dir, optimizer, epochs, features, hidden_units, train_data):
    print("Will Save Checkpoint")
    checkpoint = {'input_size': features,
                  'output_size': 102,
                  'hidden_layers': hidden_units,
                  'optimizer': optimizer.state_dict(),
                  'epochs': epochs,
                  'idx': train_data.class_to_idx,
                  'state_dict': model.state_dict()}

    with open(save_dir, 'wb') as f:
        torch.save(checkpoint, f)
    
    print("Checkpoint saved")

def main():
    args = arg_parser()
    #print(args.data_dir)
    # setup data
    trainloader, validloader, testloader, train_data = load_data(args.data_dir)
    # get general model/architecture
    if args.arch == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif args.arch == 'resnext101_32x8d':
        model = models.resnext101_32x8d(pretrained=True)
    else:
        model = models.resnet50(pretrained=True)
    ## test for others
    for param in model.parameters():
        param.requires_grad = False
    output_size = 102 #assumed from previous flowers project. could set based on number of subdirectories. maybe later
    dropout = 0.2
    model, features = create_classifier(model, args.hidden_units, dropout, output_size)
    #
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    
    #ready to train and test the model
    trained_model = train_model(model, optimizer, criterion, trainloader, validloader, args.epochs, args.gpu)
    # now validate the mode
    validate(trained_model, testloader, args.gpu)
    #save model
    save_checkpoint(trained_model, args.save_dir, optimizer, args.epochs, features, args.hidden_units, train_data)
    
    
if __name__ == '__main__': main()
