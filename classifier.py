import torch
import torch.nn as nn
from torchvision import datasets
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
import random

# Load MNIST from file
DATA_DIR = "."

train = datasets.FashionMNIST(DATA_DIR, train=True, download=False)
test = datasets.FashionMNIST(DATA_DIR, train=False, download=False)

# print(train)
# print(test)

# # Print the first 3 entries of the dataset as tuples of (image, label)
# print(train[0])
# print(train[1])
# print(train[2])

# # Print the shape of the first 3 images
# print("Shape of first entry:", train[0][0].size)
# print("Shape of second entry:", train[1][0].size)
# print("Shape of third entry:", train[2][0].size)

# Print the label of the first 3 images
# print("Label of first entry:", train[0][1])
# print("Label of first entry:", train[1][1])
# print("Label of first entry:", train[2][1])

# Store as X_train, y_train, X_test, y_test
X_train = train.data.float()
y_train = train.targets
X_test = test.data.float()
y_test = test.targets

# Print the shape of the data
def display_image(X_i, y_i):
    plt.imshow(X_i, cmap='binary')
    plt.title("Label: %d" % y_i)
    plt.show()

# Display images
# for i in range():
#display_image(X_train[7], y_train[7])

# Now let's define a function that will find an instance of a given class
def find_digit(digit, X, y):
    """
    Find an instance of a given digit in the dataset.
    :param digit: The digit to find.
    :param X: The images as a (Nx28x28) numpy array.
    :param y: The labels as a (Nx1) numpy array.
    :return: A 28x28 numpy array containing the first instance of the digit in X.
    """
    for i in range(len(y)):
        if y[i] == digit:
            return X[i]
    return None

# Find and display image with specific label
label = 5
X_digit = find_digit(label, X_train, y_train)
#display_image(X_digit, label)

# Display one image of each fashion product in a 3x4 grid
# fig, axs = plt.subplots(3, 4, figsize=(12, 12))
# for i in range(10):
#     X_digit = find_digit(i, X_train, y_train)
#     ax = axs[i//4, i%4]
#     ax.imshow(X_digit, cmap='binary')
#     ax.set_title("Label: %d" % i)
# plt.tight_layout()
# plt.show()

#Split training data into training and validation sets
# Sample random indices for validation
test_size = X_test.shape[0]
indices = np.random.choice(X_train.shape[0], test_size, replace=False)

# Create validation set
X_valid = X_train[indices]
y_valid = y_train[indices]

# Remove validation set from training set
X_train = np.delete(X_train, indices, axis=0)
y_train = np.delete(y_train, indices, axis=0)

# Print final data sizes
# print(X_train.shape)
# print(y_train.shape)
# print(X_valid.shape)
# print(y_valid.shape)
# print(X_test.shape)
# print(y_test.shape)

X_train = X_train.reshape(-1, 28*28)
X_valid = X_valid.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)

batch_size = 64

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_valid, y_valid)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define the artificial neural networks model
class ClassifierANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)

# Train the model
