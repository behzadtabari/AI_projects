# CIFAR-10 Classification with ResNet

This project demonstrates the implementation of a Residual Network (ResNet) for image classification on the CIFAR-10 dataset using PyTorch. It leverages data augmentation techniques and a custom ResNet model to improve accuracy on this image dataset.

## Description

The codebase includes several components:

1. **Data Augmentation and Preprocessing**:
   - The CIFAR-10 dataset is loaded and preprocessed with padding, random horizontal flips, and random cropping. This augmentation enhances model generalization by providing more varied data samples during training.

2. **ResNet Model**:
   - The ResNet model is built using a custom `ResidualBlock` class. Each residual block allows gradients to flow more effectively through the network, which addresses the vanishing gradient problem in deep networks.
   - The ResNet architecture used here consists of three main layers, each containing two residual blocks.

3. **Training and Testing**:
   - The training loop runs for a specified number of epochs, updating the model weights using the Adam optimizer and a cross-entropy loss function.
   - The learning rate decays every 20 epochs, reducing it by a factor of 0.5 to stabilize training over time.
   - The model is evaluated on the test set after training, and accuracy is calculated based on the percentage of correctly classified images.

### Inputs

- **Images (images)**: Batch of CIFAR-10 images with shape `(batch_size, channels, height, width)`.
- **Labels (labels)**: Class labels for each image in the batch.

### Output

The output is the model’s class predictions for each image, which are used to compute the classification accuracy on the test dataset.

### Model Hyperparameters

- **num_epochs**: Number of training epochs (default: 5)
- **batch_size**: Number of samples per batch (default: 100)
- **learning_rate (lr)**: Learning rate for the Adam optimizer (default: 0.001)

## Contact

For questions or feedback, please contact [Behzad Tabari](mailto:behzad.tabari@tum.de).
