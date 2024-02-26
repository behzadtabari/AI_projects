import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from Dataset import Dataset_modified
from Model import Model

# read the data
data = pd.read_csv('diabetes.csv')
data_test = pd.read_csv('diabetes_test.csv')

#convert data to x and y
x = data.iloc[:,0:-1].values
y_string = list(data.iloc[:,-1])

x_test = data_test.iloc[:,0:-1].values
y_string_test = list(data_test.iloc[:,-1])

#convert y_string to y_binary
y_binary = []
y_binary_test = []
for string in y_string:
    if string == "positive":
        y_binary.append(1)
    elif string == "negative":
        y_binary.append(0)
    else:
        raise Exception("output must either be positive or negative")

for string in y_string_test:
    if string == "positive":
        y_binary_test.append(1)
    elif string == "negative":
        y_binary_test.append(0)
    else:
        raise Exception("output must either be positive or negative")

# convert lists to np.array
y = np.array(y_binary, dtype = 'float64')
y_test = np.array(y_binary_test,dtype='float64')

# feature normalization, standardization
sc = StandardScaler()
x = sc.fit_transform(x)
x_test = sc.transform(x_test)
print(x_test)

#convert to numpy to pytorch
x = torch.tensor(x)
x_test = torch.tensor(x_test)
# why do we do this ??
y = torch.tensor(y).unsqueeze(1)
y_test = torch.tensor(y_test).unsqueeze(1)

#instantiate the object
dataset = Dataset_modified(x,y)
dataset_test = Dataset_modified(x_test,y_test)

# Load the data to your dataloader for batch processing and shuffling
train_loader = torch.utils.data.DataLoader(dataset = dataset,
                            batch_size = 32,
                            shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = dataset_test,
                            batch_size = 1,
                            shuffle = False)

# some minor visualization of train loader - temporary

print ("There is {} batches in the dataset".format(len(train_loader)))
for (x,y) in train_loader:
    print("for one batch, there is:")
    print("Data: {}".format(x.shape))
    print("Labels: {}".format(y.shape))
    break

# create the network
model = Model(7,1)
#Define the loss
loss_function = torch.nn.BCELoss(size_average = True)
#define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1 , momentum=0.9)


# training the network
epochs = 200
for epoch in range(epochs):
    for inputs,labels in train_loader:
        inputs = inputs.float()
        labels = labels.float()

        # forward propagation
            # this is quite intriguing that PyTorch itself calls the forward function
            # when we pass inputs to the objects, there must be some nice definition in
            # the way that functino is defined
        outputs = model.forward(inputs)
        # loss calculation
        loss = loss_function(outputs,labels)
        # clear the gradinet buffer
        optimizer.zero_grad()
        #backpropagation
        loss.backward()
        # Update the weights
        optimizer.step()
    # Accuracy Callculation
    output = (outputs > 0.5).float()
    accuracy = (output == labels).float().mean()

    # print
    print("Epoch {}/{} , loss: {:.3f}, Accuracy: {:.3f}".format(epoch+1,epochs,loss,accuracy))


# test the network
model.eval()
result = []
for inputs,labels in test_loader:
    inputs = inputs.float()
    labels = labels.float()

    # forward propagation
    results = model.forward(inputs)
    result.append((results > 0.5).float())

for res in result:
    if res == 1:
        print("positive")
    else:
        print("negative")