from sys import argv

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.datasets import ImageFolder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

classes = 4
epochs = 100
batch_size = 32
learning_rate = 0.001

transform = transforms.Compose(
    [
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
    ]
)

print("Loading dataset")
dataset_path = argv[1]
train_dataset = ImageFolder(dataset_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = ImageFolder(dataset_path, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print("Creating model")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("Training model")
for epoch in range(epochs):
    loss_avg = 0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_avg += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"\r{epoch + 1}/{epochs}: {i / len(train_loader) * 100:.0f}%", end="")

    loss_avg /= len(train_loader)
    print(f"\r{epoch + 1}/{epochs}: {loss_avg}")
    loss_avg = 0

print("Saving model")
torch.save(model.state_dict(), "resnet.pt")
