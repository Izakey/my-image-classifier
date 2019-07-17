import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
#from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse

model_architectures = {"vgg16":25088,
                       "densenet121":1024,
                       "alexnet":9216}

def load_data(where  = "./flowers"):
    '''
    Arguments : the datas' path
    Returns : The loaders for the train, validation and test datasets

    This function receives the location of the image files, applies the necessery transformations (rotations,flips,normalizations and        crops) and converts the images to tensor in order to be able to be fed into the neural network
    '''

    data_dir = where
    train_dir = "./flowers/train"
    valid_dir = "./flowers/valid"
    test_dir = "./flowers/test"
    
    color_channel_shifter_means = [0.485, 0.456, 0.406]
    color_channel_shifter_std_dev = [0.229, 0.224, 0.225]
    TRAINING_BATCH_SIZE = 64
    VALIDATION_BATCH_SIZE = TRAINING_BATCH_SIZE // 2
    TESTING_BATCH_SIZE = 20

    training_transformations = transforms.Compose([transforms.RandomRotation(30), transforms.RandomResizedCrop(224),
                                                   transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                                   transforms.Normalize(color_channel_shifter_means, color_channel_shifter_std_dev)])

    validation_transformations = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(color_channel_shifter_means, color_channel_shifter_std_dev)])

    testing_transformations = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(color_channel_shifter_means,color_channel_shifter_std_dev)])

    # DONE: Load the datasets with ImageFolder
    training_data_set = datasets.ImageFolder(train_dir, transform = training_transformations)
    validation_data_set = datasets.ImageFolder(valid_dir, transform = validation_transformations)
    testing_data_set = datasets.ImageFolder(test_dir, transform = testing_transformations)

    # DONE: Using the image datasets and the trainforms, define the dataloaders
    training_data_set_loader = torch.utils.data.DataLoader(training_data_set, batch_size = TRAINING_BATCH_SIZE, shuffle = True)
    validation_data_set_loader = torch.utils.data.DataLoader(validation_data_set, batch_size = VALIDATION_BATCH_SIZE, shuffle = True)
    testing_data_set_loader = torch.utils.data.DataLoader(testing_data_set, batch_size = TESTING_BATCH_SIZE, shuffle = True)
    
    return training_data_set_loader, validation_data_set_loader, testing_data_set_loader


def neural_network_setup(architecture='densenet121',dropout=0.5, first_hidden_layer = 120, lr = 0.001, power = 'gpu'):
    '''
    Arguments: The architecture for the network(alexnet,densenet121,vgg16), the hyperparameters for the network (hidden layer 1 nodes,      dropout and learning rate) and whether to use gpu or not
    Returns: The set up model, along with the criterion and the optimizer fo the Training
    '''
    
    if architecture == 'vgg16':
        model = models.vgg16(pretrained=True)        
    elif architecture == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif architecture == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        print("Invalid model architecture {} ! Did you mean vgg16, densenet121, or alexnet?".format(architecture))
        
    print("Len of model parameters: {}".format(len(list(model.parameters()))))
    for parameter in model.parameters():
        parameter.requires_grad = False

        classifier = nn.Sequential(OrderedDict([
            ('dropout', nn.Dropout(dropout)),
            ('inputs', nn.Linear(model_architectures[architecture], first_hidden_layer)),
            ('relu1', nn.ReLU()),
            ('first_hidden_layer', nn.Linear(first_hidden_layer, 90)),
            ('relu2', nn.ReLU()),
            ('second_hidden_layer', nn.Linear(90,80)),
            ('relu3', nn.ReLU()),
            ('third_hidden_layer', nn.Linear(80,102)),
            ('output', nn.LogSoftmax(dim=1))]))
        
        model.classifier = classifier
        criterion = nn.NLLLoss()
        print("Len of classifier parameters : {}".format(len(list(model.classifier.parameters()))))
        optimizer = torch.optim.Adam(params=model.classifier.parameters(), lr = 0.001)
        if torch.cuda.is_available() and power == 'gpu':
            model.cuda()
        
    return model , optimizer , criterion

def train_network(model, criterion, optimizer, epochs = 3, print_every=20, power = 'gpu'):
    '''
    Arguments: The model, the criterion, the optimizer, the number of epochs, the dataset, and whether to use a gpu or not
    Returns: Nothing

    This function trains the model over a certain number of epochs and displays the training,validation and accuracy every                  "print_every" step using cuda if specified. The training method is specified by the criterion and the optimizer which are NLLLoss     and Adam respectively
    '''
    steps = 0
    running_loss = 0
    first_loader, second_loader, third_loader = load_data("./flowers")

    print("--------------Training is starting------------- ")
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(first_loader):
            steps += 1
            if torch.cuda.is_available() and power == 'gpu':
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
                accuracy=0

                for ii, (inputs2,labels2) in enumerate(second_loader):
                    optimizer.zero_grad()
                    if torch.cuda.is_available():
                        inputs2, labels2 = inputs2.to('cuda:0') , labels2.to('cuda:0')
                        model.to('cuda:0')

                    with torch.no_grad():
                        outputs = model.forward(inputs2)
                        vlost = criterion(outputs,labels2)
                        ps = torch.exp(outputs).data
                        equality = (labels2.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()

                vlost = vlost / len(second_loader)
                accuracy = accuracy /len(second_loader)

                print("Epoch: {}/{}... ".format(e+1, epochs), "Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Lost {:.4f}".format(vlost), "Accuracy: {:.4f}".format(accuracy))

                running_loss = 0

    print("-------------- Finished training -----------------------")
    print("Dear User I the ulitmate NN machine trained your model. It required")
    print("----------Epochs: {}------------------------------------".format(epochs))
    print("----------Steps: {}-----------------------------".format(steps))
    print("That's a lot of steps")

def save_checkpoint(path='checkpoint.pth',architecture, first_hidden_layer=120, dropout=0.5, lr=0.001, epochs=12):
    '''
    Arguments: The saving path and the hyperparameters of the network
    Returns: Nothing

    This function saves the model at a specified by the user path

    '''
    model.class_to_idx = training_data_set.class_to_idx
    model.cpu
    torch.save({'architecture' :'densenet121', 'first_hidden_layer':120, 
                'state_dict':model.state_dict(), 'class_to_idx':model.class_to_idx},
                'checkpoint.pth')


def load_checkpoint(path='checkpoint.pth'):
    '''
    Arguments: The path of the checkpoint file
    Returns: The Neural Netowrk with all hyperparameters, weights and biases

    '''
    checkpoint = torch.load('checkpoint.pth')
    architecture = checkpoint['architecture']
    first_hidden_layer = checkpoint['first_hidden_layer']
    model,_,_ = neural_network_setup(architecture , 0.5, first_hidden_layer)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

def process_image(image_path):
    '''
    Arguments: The image's path
    Returns: The image as a tensor

    This function opens the image usign the PIL package, applies the  necessery transformations and returns the image as a tensor ready to be fed to the network

    '''
    for i in image_path:
        path = str(i)
    image_pil = Image.open(image)
   
    adjustments = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(mean=color_channel_shifter_means, std=color_channel_shifter_std_dev)])
    
    image_tensor = adjustments(image_pil)
    
    return image_tensor


def predict(image_path, model, top_K=5,power='gpu'):
    '''
    Arguments: The path to the image, the model, the number of prefictions and whether cuda will be used or not
    Returns: The "topk" most probable choices that the network predicts

    '''

    if torch.cuda.is_available() and power == 'gpu':
        model.to('cuda:0')

    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()

    if power == 'gpu':
        with torch.no_grad():
            output = model.forward(img_torch.cuda())
    else:
        with torch.no_grad():
            output=model.forward(img_torch)

    probability = F.softmax(output.data,dim=1)

    return probability.topk(top_K)