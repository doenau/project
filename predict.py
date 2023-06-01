# Predicting classes - The predict.py script successfully reads in an image and a checkpoint then prints the most likely image class and it's associated probability

# Top K classes - The predict.py script allows users to print out the top K classes along with associated probabilities

# Displaying class names - The predict.py script allows users to load a JSON file that maps the class values to other category names

# Predicting with GPU - The predict.py script allows users to use the GPU to calculate the predictions

import torch
import argparse
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import OrderedDict
import os
#from workspace_utils import active_session


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_file', help='Full path to image file', default='flowers')
    parser.add_argument('--cpt', help='Full path to checkpoint file')
    parser.add_argument('--top_k', help='# of Top Probs to show', type=int, default=5)
    parser.add_argument('--class_dict', help='Path to JSON class names', type=str, default = 'cat_to_name.json')
    parser.add_argument('--gpu', help='Use GPU', action='store_true')
    args = parser.parse_args()
    return args

def load_checkpoint(filepath):
    print("Loading Checkpoint")
    checkpoint = torch.load(filepath)
    model = models.resnet50(pretrained = True)
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(checkpoint['input_size'], checkpoint['hidden_layers'])),
                    ('relu', nn.ReLU()),
                    ('dropout', nn.Dropout(p=0.2)),
                    ('fc2', nn.Linear(checkpoint['hidden_layers'], checkpoint['output_size'])),
                    ('output', nn.LogSoftmax(dim=1))
                    ]))
    model.classifier = classifier
    model.class_to_idx = checkpoint['idx']
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model

def process_image(image):
    print("Processing Image")
    minsize = 256
    cropsize = 224
    means = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    with Image.open(image) as im:
        width, height = im.size
        if width > height:
            width, height = int(width * minsize / height), minsize
        else:
            width, height = minsize, int(height * minsize / width)
        im = im.resize([width, height])
        left = (width - cropsize) / 2
        right = (width + cropsize) / 2
        top = (height - cropsize) / 2
        bottom = (height + cropsize) / 2
        im = im.crop((left, top, right, bottom))
        im.show()
        np_image = np.array(im)
        np_image = np_image / 255.0
        np_image = (np_image - means) / std
        np_image = np_image.transpose()
        return np_image
    
def predict(image_path, model, topk=5):
    print("Running Prediction")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # TODO: Implement the code to predict the class from an image file
    img = process_image(image_path) # process the image
    
    t_img = torch.unsqueeze(torch.tensor(img),0) #convert to a tensor from a numpy array
    t_img = t_img.float()
    t_img = t_img.to(device)
    
    model.eval()
    ps = model(t_img)
    ps = torch.exp(ps)
    top_p, top_class = ps.topk(topk, dim=1)
    return top_p, top_class




#####################################
def main():
    args = arg_parser()
    # create base model
    model = models.resnet50(pretrained=True)
    # update model with saved checkpoint
    model = load_checkpoint(args.cpt)
    # get top_p and top_class
    probs, classes = predict(args.img_file, model)
    print(probs)
    print(classes)
    # get class info
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    p_flat, c_flat = probs.view(-1), classes.view(-1)
    pc_dict = {pItem.item() : cItem.item() for pItem, cItem in zip(p_flat, c_flat)}
    with open(args.class_dict, 'r') as f:
        cat_to_name = json.load(f)
    class_names = {value: cat_to_name.get(value) for value in idx_to_class.values()}
    #out = pc_dict.get(maxValue)
    #p_flat
    # show image
    im = Image.open(args.img_file)
    #display(im)
    
    #plot
    pArray = p_flat.detach().cpu().numpy()
    indices = np.arange(len(pArray))
    cArray = c_flat.detach().cpu().numpy()
    plt.barh(indices, pArray, align='center')
    plt.yticks(indices, [class_names.get(str(key)) for key in cArray])


####################################
if __name__ == '__main__': main()
