import torch
import torchvision
import torchvision.transforms as transforms

# Define transformation to be applied to the data
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize the pixel values to [-1, 1]
])

# Download and initialize the training dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)

# Download and initialize the test dataset
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)

# Define the classes
classes = tuple(str(i) for i in range(10))

# Example usage:
# Iterate through the training dataset
# for images, labels in trainloader:
#     # Training code goes here
