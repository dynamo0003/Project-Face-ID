import os
import random
from sys import argv, stdin, stdout
from time import time

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

IMAGE_SIZE = (224, 224)

# TODO add more transforms
# TODO potentially add pickle array functionality

def get_transforms(skip_extra=False):
    t = transforms
    trans = []

    if not skip_extra:
        # Flip horizontally (50% chance)
        trans.append(t.RandomHorizontalFlip())
        # Grayscale (10% chance)

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

    os.makedirs(argv[2], exist_ok=True)
    images = os.listdir(argv[1])

    count = int(argv[3])
    start_time = time()
    for i in range(count):
        image = Image.open(os.path.join(argv[1], random.choice(images)))
        image = transforms.ToTensor()(image.convert("RGB"))
        image = to_pil_image(get_transforms()(image))
        image.save(os.path.join(argv[2], f"{i}.png"))
        print(f"\r{i + 1}/{count}", end="")
        stdout.flush()

    print()
