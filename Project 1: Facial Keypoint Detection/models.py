## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # self.conv1 = nn.Conv2d(1, 32, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        # convolutional layer (sees 224x224x1 image tensor)
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        # convolutional layer (sees 112x112x8 tensor)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        # convolutional layer (sees 56x56x16 tensor)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # linear layer (32 * 28 * 28 -> 4096)
        self.fc1 = nn.Linear(32 * 28 * 28, 4096)
        # linear layer (4096 -> 512)
        self.fc2 = nn.Linear(4096, 512)
        # linear layer(512 -> 136)
        self.fc3 = nn.Linear(512, 136)
        # dropout layer (p=0.5)
        self.dropout = nn.Dropout(0.5)
        

        
    def forward(self, x):
        ## Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flatten image input
        x = x.view(x.size(0), -1)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = F.relu(self.fc2(x))
        # add dropout layer
        x = self.dropout(x)
        # add 3rd hidden layer
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
