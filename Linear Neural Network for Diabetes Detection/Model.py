import torch.nn as nn

# here in the initialization part I need to add anoter input
# a list containing the hidden layers information
class Model(nn.Module):
    def __init__(self,input_features, output_features):
        super(Model,self).__init__()
        self.fc1 = nn.Linear(input_features,5)
        self.fc2 = nn.Linear(5,4)
        self.fc3 = nn.Linear(4,3)
        self.fc4 = nn.Linear(3,output_features)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    def forward(self,x):
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.fc3(out)
        out = self.tanh(out)
        out = self.fc4(out)
        out = self.sigmoid(out)
        return out

