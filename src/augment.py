import os
import random
from sys import argv, stdin, stdout
from time import time

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms.functional as F

class CustomAffine:
    def __init__(self, translate, scale, shear):
        self. angle = 0
        self.translate = translate
        self.scale = scale
        self.shear = shear

    def __call__(self, img):
        return F.affine(img, self.angle, self.translate, self.scale, self.shear)

class CustomPerspective:
    def __init__(self, distortion_scale, zoom_factor):
        self.distortion_scale = distortion_scale
        self.zoom_factor = zoom_factor

    def __call__(self, img):
        startpoints, endpoints = self.get_params(IMAGE_SIZE, self.distortion_scale, self.zoom_factor)
        return F.perspective(img, startpoints, endpoints)

    @staticmethod
    def get_params(IMAGE_SIZE, distortion_scale, zoom_factor):
        width = IMAGE_SIZE[0]
        height = IMAGE_SIZE[1]
        half_height = width / 2
        half_width = height / 2

        # Compute the zoomed in corners
        zoomed_topleft = [half_width * (1 - 1/zoom_factor), half_height * (1 - 1/zoom_factor)]
        zoomed_topright = [width - half_width * (1 - 1/zoom_factor), half_height * (1 - 1/zoom_factor)]
        zoomed_botright = [width - half_width * (1 - 1/zoom_factor), height - half_height * (1 - 1/zoom_factor)]
        zoomed_botleft = [half_width * (1 - 1/zoom_factor), height - half_height * (1 - 1/zoom_factor)]

        # Apply distortion within the zoomed region
        topleft = [zoomed_topleft[0] + random.uniform(0, distortion_scale) * half_width,
                   zoomed_topleft[1] + random.uniform(0, distortion_scale) * half_height]
        topright = [zoomed_topright[0] - random.uniform(0, distortion_scale) * half_width,
                    zoomed_topright[1] + random.uniform(0, distortion_scale) * half_height]
        botright = [zoomed_botright[0] - random.uniform(0, distortion_scale) * half_width,
                    zoomed_botright[1] - random.uniform(0, distortion_scale) * half_height]
        botleft = [zoomed_botleft[0] + random.uniform(0, distortion_scale) * half_width,
                   zoomed_botleft[1] - random.uniform(0, distortion_scale) * half_height]

        startpoints = [[0, 0], [width, 0], [width, height], [0, height]]
        endpoints = [topleft, topright, botright, botleft]
        return startpoints, endpoints

class CustomGaussianBlur:
    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img):
        return F.gaussian_blur(img, self.kernel_size, self.sigma)

class CustomRandomNoise:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        noise = torch.randn_like(img) * self.std + self.mean
        noisy_img = img + noise
        noisy_img = torch.clamp(noisy_img, 0, 1)
        return noisy_img

IMAGE_SIZE = (224, 224)

def get_transforms(skip_extra=False):
    t = transforms
    trans = []

    if not skip_extra:
        # Flip horizontally (50% chance)
        trans.append(t.RandomHorizontalFlip())
        # Grayscale (12.5% chance)
        if random.randint(0, 7) == 0:
            trans.append(t.Grayscale(num_output_channels=1))
        # Hue (12.5% chance)
        if random.randint(0, 7) == 0:
            trans.append(t.ColorJitter(hue=0.1))
        # Brigthness, contrast, saturation (1-3 times)
        for _ in range(random.randint(0, 2)):
            if random.randint(0, 4) == 0:  # 20% chance
                trans.append(t.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
        # Crop area where size is at least 3/4 or crop around the center
        if random.randint(0, 100):
            trans.append(t.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1)))
        else:
            trans.append(t.CenterCrop(IMAGE_SIZE))
        # Rotate randomly by 0 - 20°
        trans.append(t.RandomRotation(random.randint(0, 20), fill=128))
        # Custom affine and shear (10%)
        if random.randint(0, 9) == 0:
            trans.append(CustomAffine(translate=(10, 10), scale=1, shear=random.randint(-10, 10)))
        # Custom perspective (10%) -- also zooms slightly to mitigate black borders
        if random.randint(0, 9) == 0:
            trans.append(CustomPerspective(distortion_scale=0.2, zoom_factor=0.8))
        # Custom gaussian blur (1%)
        if random.randint(0, 99) == 0:
            trans.append(CustomGaussianBlur(kernel_size=5))
        # Custom noise (1%)
        if random.randint(0, 99) == 0:
            trans.append(CustomRandomNoise(mean=0, std=0.05))

    trans.append(t.Resize(IMAGE_SIZE))
    return transforms.Compose(trans)

def augment(input_path, save_path, count): 
    images = os.listdir(input_path)

    for i in range(count):
        image = Image.open(os.path.join(input_path, random.choice(images)))
        image = transforms.ToTensor()(image.convert("RGB")).to(device)
        image = to_pil_image(get_transforms()(image))
        image.save(os.path.join(save_path, f"aug_{i:04d}.png"))
        stdout.flush()


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
