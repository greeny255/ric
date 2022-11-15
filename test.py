from __future__ import print_function, division
from torchvision import transforms
from multiprocessing import freeze_support

import os
import sys
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from SceneClassification import SceneClassificationDataset, Net, Network

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

# setting batch size
BATCH_SIZE = 4

# imgae transform
transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((300,300)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# Load data to train
dataset = SceneClassificationDataset("./hust/", ["food", "indoor", "outdoor"], transform=transform)
# suppose dataset is the variable pointing to whole datasets
N = len(dataset)
# generate & shuffle indices
indices = np.arange(N)
indices = np.random.permutation(indices)
# there are many ways to do the above two operation. (Example, using np.random.choice can be used here too

# select train/test/val, for demo I am using 70,15,15
test_indices = indices[int(0.85*N):]
test_data = data.Subset(dataset, test_indices)
testloader = data.DataLoader(test_data,
                                batch_size=BATCH_SIZE)

# Load data to train

# check device to use GPU or CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)


# get some random training images
if __name__ == '__main__':
    freeze_support()
    print('Start Testing')
    # Run Test
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # print images
    dataset.imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{dataset.classes[labels[j]]:5s}' for j in range(4)))

    # Load model
    net = Net()
    # load training result
    PATH = './classification_model.pth'
    net.load_state_dict(torch.load(PATH))

    outputs = net(images)

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in dataset.classes}
    total_pred = {classname: 0 for classname in dataset.classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[dataset.classes[label]] += 1
                total_pred[dataset.classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')




