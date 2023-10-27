import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from Model import *

input_size = 784        #Number of input neurons (image pixels)
hidden_size = 400       #Number of hidden neurons
out_size = 10        #Number of classes (0-9)
epochs = 10            #How many times we pass our entire dataset into our network
batch_size = 100        #Input size of the data during one iteration
learning_rate = 0.001   #How fast we are learning

#step zero , pre load data
train_dataset = datasets.EMNIST(root='./data',split='mnist',train=True,transform = transforms.ToTensor())
test_dataset = datasets.EMNIST(root='./root',split='mnist',train=False,transform=transforms.ToTensor())

# step one , load the data
train_loader = torch.utils.data.DataLoader(datasets = train_dataset, batch_size= batch_size, shuffle = True)
test_loader =  torch.utils.data.DataLoader(datasets = test_dataset, batch_size= batch_size, shuffle = False)

    # Create an object of the class, which represents our network


net = Net(input_size, hidden_size, out_size)
CUDA = torch.cuda.is_available()
if CUDA:
    print("CUDA is available ! ")
    net = net.cuda()



# The loss function. The Cross Entropy loss comes along with Softmax. Therefore, no need to specify Softmax as well
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
#Train the network
for epoch in range(epochs):
    correct_train = 0
    running_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        # Flatten the image from size (batch,1,28,28) --> (100,1,28,28) where 1 represents the number of channels (grayscale-->1),
        # to size (100,784) and wrap it in a variable
        images = images.view(-1, 28 * 28)
        if CUDA:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        correct_train += (predicted == labels).sum()
        loss = criterion(outputs, labels)  # Difference between the actual and predicted (loss function)
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()  # Backpropagation
        optimizer.step()  # Update the weights

    print('Epoch [{}/{}], Training Loss: {:.3f}, Training Accuracy: {:.3f}%'.format
          (epoch + 1, epochs, running_loss / len(train_loader), (100 * correct_train.double() / len(train_dataset))))
print("DONE TRAINING!")
# with torch.no_grad():
#     correct = 0
#     for images, labels in test_loader:
#         if CUDA:
#             images = images.cuda()
#             labels = labels.cuda()
#         images = images.view(-1, 28 * 28)
#         outputs = net(images)
#         _, predicted = torch.max(outputs.data, 1)
#         correct += (predicted == labels).sum().item()
#
#     print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / len(test_dataset)))
# Function to deflatten and save an image with its label
def save_image_with_label(image, label, filename):
    image = image.view(1, 28, 28)  # Reshape the flattened image to (1, 28, 28)
    vutils.save_image(image, filename)
    print(f"Saved image with label {label} as {filename}")

# Create a directory to save the images
import os
save_dir = "./saved_images/"
os.makedirs(save_dir, exist_ok=True)

# Save a few test images with their labels
num_images_to_save = 5  # You can change this to the number of images you want to save
with torch.no_grad():
    correct = 0
    for i, (images, labels) in enumerate(test_loader):
        if i >= num_images_to_save:
            break
        if CUDA:
            images = images.cuda()
            labels = labels.cuda()
        images = images.view(-1, 28 * 28)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()

        # Save the image with its label
        for j in range(len(images)):
            filename = os.path.join(save_dir, f"test_image_{i * batch_size + j}_label_{labels[j].item()}.jpg")
            save_image_with_label(images[j], labels[j], filename)

print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / len(test_dataset)))
print(f"Saved {num_images_to_save} test images with their labels in {save_dir}")