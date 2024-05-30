import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.types import Device
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.datasets import ImageFolder

transform = transforms.Compose(
    [
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
    ]
)


class Model:
    device: Device
    model: models.ResNet

    def __init__(self, classes: int = 2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, classes)
        self.model = self.model.to(self.device)

    def save(self, path: str):
        print(f"Saving trained model to {path}")
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        print(f"Loading trained model from {path}")
        self.model.load_state_dict(torch.load(path))

    def train(self, path: str, epochs: int, batch_size: int, learning_rate: float):
        print(f"Training model")
        self.model.train()

        dataset = ImageFolder(path, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            loss_avg = 0

            for i, (images, labels) in enumerate(loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss_avg += loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f"\r[{epoch + 1}/{epochs}]: {i / len(loader) * 100:.0f}%", end="")

            loss_avg /= len(loader)
            print(f"\r[{epoch + 1}/{epochs}]: {loss_avg}")
            if round(loss_avg, 1) <= 0.1:
                break
            loss_avg = 0

    def eval(self, path: str):
        print("Evaluating image")
        self.model.eval()

        with torch.no_grad():
            image = Image.open(path).convert("RGB")
            image = transforms.ToTensor()(image).unsqueeze(0).to(self.device)
            print(self.model(image))
