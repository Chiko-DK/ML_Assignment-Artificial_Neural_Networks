#%matplotlib inline
from torchvision import datasets
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
import random

# Load MNIST from file
DATA_DIR = "."
download_dataset = False

train_mnist = datasets.FashionMNIST(DATA_DIR, train=True, download=False)
test_mnist = datasets.FashionMNIST(DATA_DIR, train=False, download=False)

print(train_mnist)
print(test_mnist)
