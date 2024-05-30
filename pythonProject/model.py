from sys import argv

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.datasets import ImageFolder

transform = transforms.Compose(
    [
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
    ]
)


class Model:
    def __init__(self, classes: int = 2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, classes)
        self.model = self.model.to(self.device)

    def save(self, path: str):
        print("Saving model")
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        pass

    def train(self, path: str, epochs: int, batch_size: int, learning_rate: float):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        dataset = ImageFolder(path, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        print("Training model")
        for epoch in range(epochs):
            loss_avg = 0

            for i, (images, labels) in enumerate(loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss_avg += loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f"\r{epoch + 1}/{epochs}: {i / len(loader) * 100:.0f}%", end="")

            loss_avg /= len(loader)
            print(f"\r{epoch + 1}/{epochs}: {loss_avg}")
            loss_avg = 0

    def eval(self):
        pass


model = Model(4)
model.train(argv[1], 3, 32, 0.001)
