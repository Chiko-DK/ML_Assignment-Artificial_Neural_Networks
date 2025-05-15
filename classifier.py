import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision import datasets
from PIL import Image
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
import random
import os

# Load MNIST from file
DATA_DIR = "."

train = datasets.FashionMNIST(DATA_DIR, train=True, download=False)
test = datasets.FashionMNIST(DATA_DIR, train=False, download=False)

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

X_train = X_train.reshape(-1, 28*28)
X_valid = X_valid.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)

X_train = X_train / 255.0
X_valid = X_valid / 255.0
X_test = X_test / 255.0

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
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)

print("Training the model...")
print("...")
# Train the model
model = ClassifierANN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

EPOCHS = 30

for epoch in range(EPOCHS):
    # Training phase
    model.train()
    train_loss = 0.0
    train_correct = 0

    for X_batch, y_batch in train_loader:
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X_batch.size(0)
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == y_batch).sum().item()

    train_loss /= len(train_loader.dataset)
    train_accuracy = train_correct / len(train_loader.dataset)

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            val_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == y_batch).sum().item()

    val_loss /= len(val_loader.dataset)
    val_accuracy = val_correct / len(val_loader.dataset)

    # print(f"Epoch {epoch+1}/{EPOCHS} "
    #       f"| Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} "
    #      f"| Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
    print(f"--- Training Progress: {int(((epoch+1)/EPOCHS)*100)}% ---")

# Compute accuracy
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# # Display a few images and their predictions
# model.eval()
# with torch.no_grad():
#     for i in range(10):
#         idx = random.randint(0, X_test.shape[0])
#         image = X_test[idx]
#         label = y_test[idx]
#         output = model(image.clone().detach().view(1, -1))
#         _, predicted = torch.max(output, 1)
#         print("True label: %d, Predicted label: %d" % (label, predicted))
#         # display_image(image.reshape(28, 28), label)


# Mapping from class index to label
class_labels = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

print("Done!")

print("Please enter a filepath:")
while True:
    user_input = input("> ")
    if user_input.lower() == "exit":
        print("Exiting...")
        break

    if not os.path.isfile(user_input):
        print("File not found. Please try again.")
        continue

    try:
        # Load image using torchvision (as grayscale)
        img = torchvision.io.read_image(user_input, mode=torchvision.io.ImageReadMode.GRAY)
        img = img.squeeze()         # Convert shape (1, 28, 28) → (28, 28)

        if img.shape != (28, 28):
            img_pil = transforms.ToPILImage()(img.unsqueeze(0))
            img_pil = img_pil.resize((28, 28))
            img = transforms.ToTensor()(img_pil).squeeze() * 255.0

        img = img.float().reshape(1, 28*28) / 255.0

        model.eval()
        with torch.no_grad():
            output = model(img.view(1, -1))
            _, predicted = torch.max(output, 1)
            print("Classifier:", class_labels[predicted])
    except Exception as e:
        print(f"Error processing image: {e}")