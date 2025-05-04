
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

# Convert to numpy arrays
X_train = X_train.numpy()
y_train = y_train.numpy()
X_test = X_test.numpy()
y_test = y_test.numpy()

# print(X_train[0])
# print(y_train[0])

def display_image(X_i, y_i):
    plt.imshow(X_i, cmap='binary')
    plt.title("Label: %d" % y_i)
    plt.show()

# Display images
# for i in range():
#display_image(X_train[7], y_train[7])

# Now let's define a function that will find an instance of a given digit.
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

X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)





