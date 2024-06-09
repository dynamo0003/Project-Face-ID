import subprocess
import pytest
import sys
from PIL import Image
from torchvision.transforms import ToTensor
sys.path.append("src")
from augment import CustomAffine, CustomPerspective, CustomGaussianBlur, CustomRandomNoise, get_transforms


# Define a fixture to create a sample image tensor
@pytest.fixture
def sample_image():
    # Create a sample image with random values
    image = Image.new("RGB", (224, 224), color="white")
    return ToTensor()(image)


def test_custom_affine(sample_image):
    transform = CustomAffine(translate=(10, 10), scale=1, shear=10)
    transformed_image = transform(sample_image)
    assert transformed_image.shape == sample_image.shape, "CustomAffine transformation failed."


def test_custom_perspective(sample_image):
    transform = CustomPerspective(distortion_scale=0.2, zoom_factor=0.8)
    transformed_image = transform(sample_image)
    assert transformed_image.shape == sample_image.shape, "CustomPerspective transformation failed."


def test_custom_gaussian_blur(sample_image):
    transform = CustomGaussianBlur(kernel_size=5)
    transformed_image = transform(sample_image)
    assert transformed_image.shape == sample_image.shape, "CustomGaussianBlur transformation failed."


def test_custom_random_noise(sample_image):
    transform = CustomRandomNoise(mean=0, std=0.05)
    transformed_image = transform(sample_image)
    assert transformed_image.shape == sample_image.shape, "CustomRandomNoise transformation failed."


def test_get_transforms(sample_image):
    transform = get_transforms(skip_extra=True)
    transformed_image = transform(sample_image)
    assert transformed_image.shape == sample_image.shape, "get_transforms function failed."

if __name__ == "__main__":
    pytest.main()