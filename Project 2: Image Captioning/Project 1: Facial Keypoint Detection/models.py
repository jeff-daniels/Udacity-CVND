## define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        '''
        Modification of LeNet, fast and simple.
        input shape is 1 x 96 x 96
        output shape is 136
        '''
        ## Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # Convolutional Output size = (W-F+2P)/S + 1
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        
        # convolutional layer (1 x 96 x 96 -> 6 x 92 x 92)
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(p=0.1)
        
        # convolutional layer (6 x 46 x 46 -> 16 x 42 x 42)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(p=0.1)
        
        # linear layer (16 x 21 x 21 -> 7056)
        self.fc1 = nn.Linear(7056, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.dropout3 = nn.Dropout(p=0.1)
        
        # linear layer (512 -> 136)
        self.fc2 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.dropout4 = nn.Dropout(p=0.1)
        
        # linear layer(512 -> 136)
        self.fc3 = nn.Linear(512 ,136)
        
        # initialize weights
        I.xavier_uniform_(self.conv1.weight)
        I.xavier_uniform_(self.conv2.weight)
        I.xavier_uniform_(self.fc1.weight)
        I.xavier_uniform_(self.fc2.weight)
        I.xavier_uniform_(self.fc3.weight)
        
        
    def forward(self, x):
        ## Define the feedforward behavior of this model
        
        x = self.dropout1(self.pool1(F.elu(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(F.elu(self.bn2(self.conv2(x)))))
        
        # flatten image input
        x = x.view(x.size(0), -1)
        
        x = self.dropout3(F.elu(self.bn3(self.fc1(x))))
        x = self.dropout4(F.elu(self.bn4(self.fc2(x))))

        x = self.fc3(x)
        
        return x

class NaimishNet(nn.Module):

    def __init__(self):
        super(NaimishNet, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # Convolutional Output size = (W-F+2P)/S + 1
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        
        # convolutional layer (1 x 96 x 96 -> 32 x 93 x 93)        
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2,2)
        self.dropout1 = nn.Dropout(p=0.1)

        # convolutional layer (32 x 46 x 46 -> 64 x 44 x 44)    
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2,2)
        self.dropout2 = nn.Dropout(p=0.2)
        
        # convolutional layer (64 x 22 x 22 -> 128 x 21 x 21)
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2,2)
        self.dropout3 = nn.Dropout(p=0.3)

        # convolutional layer (128 x 10 x 10 -> 256 x 10 x 10)
        self.conv4 = nn.Conv2d(128, 256, 1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2,2)
        self.dropout4 = nn.Dropout(p=0.4)

        # fully connected layer (256*5*5 = 6400 -> 1000)
        self.fc1 = nn.Linear(6400, 1000)
        self.bn5 = nn.BatchNorm1d(1000)
        self.dropout5 = nn.Dropout(p=0.5)

        # fully connected layer (1000 -> 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.bn6 = nn.BatchNorm1d(1000)
        self.dropout6 = nn.Dropout(p=0.6)

        # fully connected layer (1000 -> 136)
        self.fc3 = nn.Linear(1000, 136)
        
        # initialize weights
        I.xavier_uniform_(self.conv1.weight)
        I.xavier_uniform_(self.conv2.weight)
        I.xavier_uniform_(self.fc1.weight)
        I.xavier_uniform_(self.fc2.weight)
        I.xavier_uniform_(self.fc3.weight)
        
    def forward(self, x):
        ## Define the feedforward behavior of this model
      
        x = self.dropout1(self.pool1(F.elu(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(F.elu(self.bn2(self.conv2(x)))))
        x = self.dropout3(self.pool3(F.elu(self.bn3(self.conv3(x)))))
        x = self.dropout4(self.pool4(F.elu(self.bn4(self.conv4(x)))))

        # flatten
        x = x.view(x.size(0),-1)

        x = self.dropout5(F.elu(self.bn5(self.fc1(x))))
        x = self.dropout6(F.elu(self.bn6(self.fc2(x))))
        x = self.fc3(x)

        return x
    
class LargeNaimishNet(nn.Module):

    def __init__(self):
        super(LargeNaimishNet, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # Convolutional Output size = (W-F+2P)/S + 1
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        
        # convolutional layer (1 x 224 x 224 -> 32 x 221 x 221)        
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2,2)
        self.dropout1 = nn.Dropout(p=0.1)

        # convolutional layer (32 x 110 x 110 -> 64 x 108 x 108)    
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2,2)
        self.dropout2 = nn.Dropout(p=0.2)
        
        # convolutional layer (64 x 54 x 54 -> 128 x 53 x 53)
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2,2)
        self.dropout3 = nn.Dropout(p=0.3)

        # convolutional layer (128 x 26 x 26 -> 256 x 26 x 26)
        self.conv4 = nn.Conv2d(128, 256, 1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2,2)
        self.dropout4 = nn.Dropout(p=0.4)

        # fully connected layer (256*13*13 = 43264 -> 1000)
        self.fc1 = nn.Linear(43264, 1000)
        self.bn5 = nn.BatchNorm1d(1000)
        self.dropout5 = nn.Dropout(p=0.5)

        # fully connected layer (1000 -> 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.bn6 = nn.BatchNorm1d(1000)
        self.dropout6 = nn.Dropout(p=0.6)

        # fully connected layer (1000 -> 136)
        self.fc3 = nn.Linear(1000, 136)
        
        # initialize weights
        I.xavier_uniform_(self.conv1.weight)
        I.xavier_uniform_(self.conv2.weight)
        I.xavier_uniform_(self.fc1.weight)
        I.xavier_uniform_(self.fc2.weight)
        I.xavier_uniform_(self.fc3.weight)
        
    def forward(self, x):
        ## Define the feedforward behavior of this model
      
        x = self.dropout1(self.pool1(F.elu(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(F.elu(self.bn2(self.conv2(x)))))
        x = self.dropout3(self.pool3(F.elu(self.bn3(self.conv3(x)))))
        x = self.dropout4(self.pool4(F.elu(self.bn4(self.conv4(x)))))

        # flatten
        x = x.view(x.size(0),-1)

        x = self.dropout5(F.elu(self.bn5(self.fc1(x))))
        x = self.dropout6(F.elu(self.bn6(self.fc2(x))))
        x = self.fc3(x)

        return x