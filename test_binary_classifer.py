import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import binary_classifier_r3d_input


# Create random input data and labels
batch_size = 32
input_shape = (batch_size, 512, 4, 7, 7)
labels_shape = (batch_size, 1)
input_data = torch.randn(input_shape)
labels = torch.randint(0, 2, labels_shape).float()

# Create DataLoader for training and testing
dataset = torch.utils.data.TensorDataset(input_data, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create an instance of the BinaryClassifier
model = BinaryClassifier()

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {running_loss / len(dataloader)}")

# Testing
test_input_data = torch.randn(input_shape)
test_labels = torch.randint(0, 2, labels_shape).float()
test_dataset = torch.utils.data.TensorDataset(test_input_data, test_labels)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, targets in test_dataloader:
        outputs = model(inputs)
        predicted = torch.round(outputs)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100}%")
