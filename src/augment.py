import os
import random
from sys import argv, stdin, stdout
from time import time

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

class CustomTransform:
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

    def __call__(self, img):
        # TODO LOGIC
        # For demonstration purposes
        print(f"Applying custom transform with param1: {self.param1} and param2: {self.param2}")
        return img

IMAGE_SIZE = (224, 224)

# TODO add more transforms
# TODO potentially add pickle array functionality

custom_transform = CustomTransform(param1=10, param2=20)
transform = transforms.Compose([
    custom_transform,
    transforms.ToTensor()
])

def get_transforms(skip_extra=False):
    t = transforms
    trans = []

    if not skip_extra:
        # Flip horizontally (50% chance)
        trans.append(t.RandomHorizontalFlip())
        # Grayscale (12.5% chance)
        if not random.randint(0, 7) == 0:
            trans.append(t.Grayscale(num_output_channels=1))
        # Hue (12.5% chance)
        if random.randint(0, 7) == 0:
            trans.append(t.ColorJitter(hue=0.1))
        # Brigthness, contrast, saturation (1-3 times)
        for _ in range(random.randint(0, 2)):
            if random.randint(0, 4) == 0:  # 20% chance
                trans.append(t.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3))
        # Crop area where size is at least 3/4 or crop around the center
        if random.randint(0, 100):
            trans.append(t.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1)))
        else:
            trans.append(t.CenterCrop(IMAGE_SIZE))
        # Rotate randomly by 0 - 20°
        trans.append(t.RandomRotation(random.randint(0, 20), fill=128))
        # Custom transform test (12.5%)
        if random.randint(0, 7) == 0:
            trans.append(CustomTransform(custom_transform.param1, custom_transform.param2))


    trans.append(t.Resize(IMAGE_SIZE))
    return transforms.Compose(trans)


if __name__ == "__main__":
    if len(argv) < 4:
        print(f"Usage: {argv[0]} <input path> <save path> <count> [seed]")
        exit(1)
    if len(argv) >= 5:
        seed = int(argv[4])
        random.seed(seed)
        torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(argv[2], exist_ok=True)
    images = os.listdir(argv[1])

    count = int(argv[3])
    start_time = time()
    for i in range(count):
        image = Image.open(os.path.join(argv[1], random.choice(images)))
        image = transforms.ToTensor()(image.convert("RGB")).to(device)
        image = to_pil_image(get_transforms()(image))
        image.save(os.path.join(argv[2], f"{i}.png"))
        print(f"\r{i + 1}/{count}", end="")
        stdout.flush()

    minutes, seconds = map(int, divmod(time() - start_time, 60))
    print(f"\nTook {minutes}:{seconds:02}")
    print()
