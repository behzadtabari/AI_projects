import torch.nn as nn

class CNN(nn.Module):
    def __init__(self,input_size,output_size,batch_size):
        super(CNN,self).__init__()
        # the kernel size meets the same padding situation
        self.cnn1 = nn.Conv2d(in_channels=input_size,out_channels=8,kernel_size=3,padding=1,stride=1)
        #give the number of channels to the BatchNorm2d
        self.batchnorm1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size = 2)
        # the kernel size meets the same padding situation
        self.cnn2 = nn.Conv2d(in_channels=8,out_channels=32,kernel_size=5,padding=2,stride=1)
        #give the number of channels to the BatchNorm2d
        self.batchnorm2 = nn.BatchNorm2d(32)
        # the input of the first linear layer would be : 32 * ((28/2)/2) * ((28/2)/2) = 1568
        self.fc1 = nn.Linear(1568, 600)
        self.dropout = nn.Dropout(p = 0.5)
        self.fc2 = nn.Linear(600, output_size)
        self.batch_size = batch_size


    def forward(self,x):
        output = self.cnn1(x)
        output = self.batchnorm1(output)
        output = self.relu(output)
        output = self.maxpool(output)
        output = self.cnn2(output)
        output = self.batchnorm2(output)
        output = self.relu(output)
        output = self.maxpool(output)
        # the output of maxpool is yet 2D and must be flattened
        output = output.view(-1,1568)
        output = self.fc1(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        return output