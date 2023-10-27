import cv2
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import random
import torch.utils.data as data
from Model import CNN


# Predict your own image


CUDA = torch.cuda.is_available()
if CUDA:
    print("GPU is available")
else:
    print("GPU is not available")
# data
gray_mean = 0.1307
gray_std = 0.3081
batch_size = 100
num_epochs = 20

#transfroms
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((gray_mean,),(gray_std,))])

transforms_photo = transforms.Compose([transforms.Resize((28,28)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((gray_mean,), (gray_std,))])

def predict(img_name, model):
    image = cv2.imread(img_name, 0)  # Read the image
    ret, thresholded = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)  # Threshold the image
    img = 255 - thresholded  # Apply image negative
    cv2.imshow('Original', img)  # Display the processed image

    # Add a loop to keep the window open until a key is pressed
    while True:
        key = cv2.waitKey(1)  # Wait for a key press for 1 ms
        if key != -1:
            break

    cv2.destroyAllWindows()
    img = Image.fromarray(img)  # Convert the image to an array
    img = transforms_photo(img)  # Apply the transformations
    img = img.view(1, 1, 28, 28)  # Add batch size

    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        img = img.cuda()

    output = model(img)
    print(output)
    print(output.data)
    _, predicted = torch.max(output, 1)
    return predicted.item()

# import dataset

train_dataset = datasets.MNIST("./data", transform=transform,train=True,download=True)
test_dataset = datasets.MNIST("./data", transform=transform,train=False)


# check the train data randomly, to do it just uncomment the following block
'''
rand_int = train_dataset[random.randint(1,60000)]
rand_image = rand_int[0].numpy() * gray_std + gray_mean
print("the label of the shown image is: ", rand_int[1])
plt.imshow(rand_image.reshape(28,28), cmap= 'gray')
plt.show()
'''
# load the datasets
train_loader = data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle=True)
test_loader = data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle=False)

# create the model and move the object to GPU
model = CNN(1,10,batch_size)
model = model.cuda()

# define the loss function and the optimization technique
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.01)

#uncomment this if you want to understand what is going on in every batch operation
'''
for i,(inputs,labels) in enumerate (train_loader):
    inputs = inputs.cuda()
    labels = labels.cuda()

    print("for one iteration in a batch: ")
    print("inputs shape: ", inputs.shape)
    print("[batch size, number of channels, height, weight]")
    print("labels shape:", labels.shape)
    output = model.forward(inputs)
    print("the output shape is deduced from the model output shape + number of batches")
    print("Output shape: ", output.shape)
    print("output Tensor:")
    print(output)
    _, predicted_nodata = torch.max(output,1)
    print("Predicted Shape", predicted_nodata.shape)
    print("Predicted Tensor:")
    print(predicted_nodata)
    break
'''

#training
train_loss = []
train_accuracy = []
test_loss = []
test_accuracy = []

for i in range(num_epochs):
    correct = 0
    iter = 0
    iter_loss = 0.0
    print( "epoch number: ", i+1)
    # set the model in testing mode
    model.train( )
    for i,(input,label) in enumerate (train_loader):
        input = input.cuda()
        label = label.cuda()

        output = model.forward(input  )
        loss = loss_function(output,label)
        iter_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(output,1)
        correct += (predicted == label).sum().item()
        iter += 1
    print("train loss at this epoch: {:.3f}".format(iter_loss/iter ))
    print("train accuracy at this epoch: {:.3f}".format( correct/len(train_dataset)))
    train_loss.append(iter_loss/iter)
    train_accuracy.append(correct/len(train_dataset))

    correct = 0
    iter = 0
    test_iter_loss = 0.0

    #set the model to testing mode
    model.eval()

    for i,(input,label) in enumerate (test_loader):
        input = input.cuda()
        label = label.cuda()

        output = model.forward(input  )
        loss = loss_function(output,label)
        test_iter_loss += loss.item()

        _, predicted = torch.max(output,1)
        correct += (predicted == label).sum().item()
        iter += 1
    print("test loss at this epoch: {:.3f}".format(test_iter_loss/iter ))
    print("test accuracy at this epoch: {:.3f}".format( correct/len(test_dataset)))
    test_loss.append(test_iter_loss/iter)
    test_accuracy.append(correct/len(test_dataset))

#printing the loss
f = plt.figure(figsize=(10,10))
plt.plot(train_loss, label = "train loss")
plt.plot(test_loss, label = "test loss")
plt.legend()
plt.show()

#printing the accuracy
g = plt.figure(figsize=(10,10))
plt.plot(train_accuracy, label = "train accuracy")
plt.plot(test_accuracy, label = "test accuracy")
plt.legend()
plt.show()


# #predcting an image
# rand_int_image = train_dataset[random.randint(1,10000)]
# rand_image = test_dataset[rand_int_image][0].resize_((1,1,28,28))
# rand_image_label = test_dataset[rand_image][1]
#
# model.eval()
#
# if CUDA:
#     model = model.cuda()
#     rand_image = rand_image.cuda()
#
# output = model.forward(rand_image)
# _, predicted = torch.max(output,1)
#
# print ( "predicted label is {}".format(predicted.item()))
# print("the label is {}".format(rand_image_label))

#predict a handwritten image
pred = predict('./data/0.jpg', model)
print("The Predicted Label is {}".format(pred))
pred = predict('./data/1.jpg', model)
print("The Predicted Label is {}".format(pred))
pred = predict('./data/2.jpg', model)
print("The Predicted Label is {}".format(pred))
pred = predict('./data/3.jpg', model)
print("The Predicted Label is {}".format(pred))
pred = predict('./data/4.jpg', model)
print("The Predicted Label is {}".format(pred))
pred = predict('./data/5.jpg', model)
print("The Predicted Label is {}".format(pred))
pred = predict('./data/6.jpg', model)
print("The Predicted Label is {}".format(pred))
pred = predict('./data/7.jpg', model)
print("The Predicted Label is {}".format(pred))
pred = predict('./data/8.jpg', model)
print("The Predicted Label is {}".format(pred))
pred = predict('./data/9.jpg', model)
print("The Predicted Label is {}".format(pred))

