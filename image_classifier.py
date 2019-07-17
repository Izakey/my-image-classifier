"""
Imports Here
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from PIL import Image
import json
from collections import OrderedDict

"""
Load The Data
"""

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

"""
Define transforms such as random scaling, rotation, cropping
 flipping for the training, validation, and testing sets
"""

color_channel_shifter_means = [0.485, 0.456, 0.406]
color_channel_shifter_std_dev = [0.229, 0.224, 0.225]

TRAINING_BATCH_SIZE = 64
VALIDATION_BATCH_SIZE = TRAINING_BATCH_SIZE // 2
TESTING_BATCH_SIZE = 20

training_transformations = transforms.Compose(
                                [transforms.RandomRotation(30), transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                 transforms.Normalize(color_channel_shifter_means, color_channel_shifter_std_dev)])

validation_transformations = transforms.Compose(
                                [transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize(color_channel_shifter_means, color_channel_shifter_std_dev)])

testing_transformations = transforms.Compose(
                                [transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize(color_channel_shifter_means, color_channel_shifter_std_dev)])

# Load the data-sets with ImageFolder
training_data_set = datasets.ImageFolder(train_dir, transform=training_transformations)
validation_data_set = datasets.ImageFolder(valid_dir, transform=validation_transformations)
testing_data_set = datasets.ImageFolder(test_dir, transform=testing_transformations)

# Using the image data-sets and the transforms, define data loaders to load data
training_data_set_loader = torch.utils.data.DataLoader(training_data_set,
                                                       batch_size = TRAINING_BATCH_SIZE,
                                                       shuffle = True)
validation_data_set_loader = torch.utils.data.DataLoader(validation_data_set,
                                                         batch_size = VALIDATION_BATCH_SIZE,
                                                         shuffle = True)
testing_data_set_loader = torch.utils.data.DataLoader(testing_data_set,
                                                      batch_size = TESTING_BATCH_SIZE,
                                                      shuffle = True)

"""
Build And Train The Network
"""

# DONE: Build and train your network
model_architectures = {"vgg16":25088,
                       "densenet121" : 1024,
                       "alexnet" : 9216 }


def neural_network_setup(architecture='vgg16', dropout=0.5, first_hidden_layer=120, lr=0.001):
    if architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif architecture == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif architecture == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        print("Invalid model architecture {} ! Did you mean vgg16, densenet121, or alexnet?".format(architecture))

    for parameter in model.parameters():
        parameter.requires_grad = False

        classifier = nn.Sequential(OrderedDict([
            ('dropout', nn.Dropout(dropout)),
            ('inputs', nn.Linear(model_architectures[architecture], first_hidden_layer)),
            ('relu1', nn.ReLU()),
            ('first_hidden_layer', nn.Linear(first_hidden_layer, 90)),
            ('relu2', nn.ReLU()),
            ('second_hidden_layer', nn.Linear(90, 80)),
            ('relu3', nn.ReLU()),
            ('third_hidden_layer', nn.Linear(80, 102)),
            ('output', nn.LogSoftmax(dim=1))]))

        model.classifier = classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr)
        model.cuda()

    return model, optimizer, criterion

model, optimizer, criterion = neural_network_setup('densenet121')

epochs = 12
print_every = 5
steps = 0
loss_show = []

# change to cuda
model.to('cuda')

for e in range(epochs):
    running_loss = 0
    for ii, (inputs, labels) in enumerate(training_data_set_loader):
        steps += 1

        inputs, labels = inputs.to('cuda'), labels.to('cuda')

        optimizer.zero_grad()

        # Forward and backward passes
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            model.eval()
            vlost = 0
            accuracy = 0

            for ii, (inputs2, labels2) in enumerate(validation_data_set_loader):
                optimizer.zero_grad()

                inputs2, labels2 = inputs2.to('cuda:0'), labels2.to('cuda:0')
                model.to('cuda:0')
                with torch.no_grad():
                    outputs = model.forward(inputs2)
                    vlost = criterion(outputs, labels2)
                    ps = torch.exp(outputs).data
                    equality = (labels2.data == ps.max(1)[1])
                    accuracy += equality.type_as(torch.FloatTensor()).mean()

            vlost = vlost / len(validation_data_set_loader)
            accuracy = accuracy / len(validation_data_set_loader)

            print("Epoch: {}/{}... ".format(e + 1, epochs),
                  "Loss: {:.4f}".format(running_loss / print_every),
                  "Validation Lost {:.4f}".format(vlost),
                  "Accuracy: {:.4f}".format(accuracy))

            running_loss = 0

"""
Testing Your Network
"""


# DONE: Do validation on the test set
def check_accuracy_on_test(testing_data_set_loader):
    correct = 0
    total = 0
    model.to('cuda:0')
    with torch.no_grad():
        for data in testing_data_set_loader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the neural network on test data set of images: %d %%' % (100 * correct / total))

check_accuracy_on_test(testing_data_set_loader)



