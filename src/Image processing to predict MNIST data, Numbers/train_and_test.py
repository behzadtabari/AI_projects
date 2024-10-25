import cv2
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import random
import torch.utils.data as data
from model import CNN 

"""
Predict your own image.

My handwritten numbers are in the data folder. Feel free to try it for yourself.
"""

cuda_available = torch.cuda.is_available()
if cuda_available:
    print("GPU is available")
else:
    print("GPU is not available")

# Data settings
GRAY_MEAN = 0.1307
GRAY_STD = 0.3081
BATCH_SIZE = 100
NUM_EPOCHS = 20

# Transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((GRAY_MEAN,), (GRAY_STD,))
])

transform_photo = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((GRAY_MEAN,), (GRAY_STD,))
])

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
    img = transform_photo(img)  # Apply the transformations
    img = img.view(1, 1, 28, 28)  # Add batch size

    model.eval()

    if cuda_available:
        model = model.cuda()
        img = img.cuda()

    output = model(img)
    print(output)
    print(output.data)
    _, predicted = torch.max(output, 1)
    return predicted.item()

# Import dataset
train_dataset = datasets.MNIST(
    "./data", transform=transform, train=True, download=True
)
test_dataset = datasets.MNIST(
    "./data", transform=transform, train=False
)

# Check the train data randomly
'''
rand_int = train_dataset[random.randint(1, 60000)]
rand_image = rand_int[0].numpy() * GRAY_STD + GRAY_MEAN
print("The label of the shown image is:", rand_int[1])
plt.imshow(rand_image.reshape(28, 28), cmap='gray')
plt.show()
'''

# Load the datasets
train_loader = data.DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
test_loader = data.DataLoader(
    dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False
)

# Create the model and move it to GPU
model = CNN(1, 10, BATCH_SIZE)
model = model.cuda()

# Define the loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Uncomment for understanding batch operations
'''
for i, (inputs, labels) in enumerate(train_loader):
    inputs = inputs.cuda()
    labels = labels.cuda()

    print("For one iteration in a batch:")
    print("Inputs shape:", inputs.shape)
    print("[Batch size, number of channels, height, weight]")
    print("Labels shape:", labels.shape)
    
    output = model.forward(inputs)
    print("Output shape:", output.shape)
    print("Output tensor:")
    print(output)
    
    _, predicted_nodata = torch.max(output, 1)
    print("Predicted shape:", predicted_nodata.shape)
    print("Predicted tensor:")
    print(predicted_nodata)
    break
'''

# Training
train_loss = []
train_accuracy = []
test_loss = []
test_accuracy = []

for epoch in range(NUM_EPOCHS):
    correct = 0
    total_loss = 0.0
    print(f"Epoch number: {epoch + 1}")

    # Training mode
    model.train()

    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()

        output = model(inputs)
        loss = loss_function(output, labels)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(output, 1)
        correct += (predicted == labels).sum().item()

    print(f"Train loss at this epoch: {total_loss / len(train_loader):.3f}")
    print(f"Train accuracy at this epoch: {correct / len(train_dataset):.3f}")
    train_loss.append(total_loss / len(train_loader))
    train_accuracy.append(correct / len(train_dataset))

    correct = 0
    total_test_loss = 0.0

    # Set model to evaluation mode
    model.eval()

    for inputs, labels in test_loader:
        inputs = inputs.cuda()
        labels = labels.cuda()

        output = model(inputs)
        loss = loss_function(output, labels)
        total_test_loss += loss.item()

        _, predicted = torch.max(output, 1)
        correct += (predicted == labels).sum().item()

    print(f"Test loss at this epoch: {total_test_loss / len(test_loader):.3f}")
    print(f"Test accuracy at this epoch: {correct / len(test_dataset):.3f}")
    test_loss.append(total_test_loss / len(test_loader))
    test_accuracy.append(correct / len(test_dataset))

# Plotting the loss
plt.figure(figsize=(10, 10))
plt.plot(train_loss, label="Train Loss")
plt.plot(test_loss, label="Test Loss")
plt.legend()
plt.show()

# Plotting the accuracy
plt.figure(figsize=(10, 10))
plt.plot(train_accuracy, label="Train Accuracy")
plt.plot(test_accuracy, label="Test Accuracy")
plt.legend()
plt.show()

# Predict a handwritten image
pred = predict('./data/0.jpg', model)
print(f"The Predicted Label is {pred}")

# You can add more predictions for other images similarly...
