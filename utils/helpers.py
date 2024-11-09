import numpy as np
import cv2
import albumentations as A
import streamlit as st

def apply_noise(image, noise_type='gaussian', noise_level=0.05):
    """
    Apply noise to an image.

    Parameters:
        image (np.ndarray): The input image in RGB format.
        noise_type (str): Type of noise to apply ('gaussian' or 'salt_pepper').
        noise_level (float): The amount of noise to apply.

    Returns:
        np.ndarray: The noisy image in RGB format.
    """
    if noise_type == 'gaussian':
        row, col, ch = image.shape
        mean = 0
        sigma = noise_level * 255
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        noisy = image + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)
    elif noise_type == 'salt_pepper':
        s_vs_p = 0.5
        amount = noise_level
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        for c in range(image.shape[2]):
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]]
            out[coords[0], coords[1], c] = 255

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p))
        for c in range(image.shape[2]):
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]]
            out[coords[0], coords[1], c] = 0
        return out
    return image

def apply_salt_pepper_noise(image, noise_level=0.05):
    """
    Apply salt and pepper noise to the image.

    Parameters:
        image (np.ndarray): The input image in RGB format.
        noise_level (float): The amount of noise to apply.

    Returns:
        np.ndarray: The noisy image in RGB format.
    """
    return apply_noise(image, noise_type='salt_pepper', noise_level=noise_level)

def apply_sobel_edge_detection(image):
    """
    Apply Sobel edge detection to the image.

    Parameters:
        image (np.ndarray): The input image in RGB format.

    Returns:
        np.ndarray: The image with Sobel edges detected in RGB format.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = np.uint8(np.clip(sobel, 0, 255))
    sobel_rgb = cv2.cvtColor(sobel, cv2.COLOR_GRAY2RGB)
    return sobel_rgb

def apply_laplacian_edge_detection(image):
    """
    Apply Laplacian edge detection to the image.

    Parameters:
        image (np.ndarray): The input image in RGB format.

    Returns:
        np.ndarray: The image with Laplacian edges detected in RGB format.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    laplacian_rgb = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2RGB)
    return laplacian_rgb

def apply_vertical_flip(image):
    """Flip the image vertically."""
    return cv2.flip(image, 0)

def apply_horizontal_flip(image):
    """Flip the image horizontally."""
    return cv2.flip(image, 1)

def apply_rotation(image, angle):
    """Rotate the image by the specified angle."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def apply_scaling(image, scale_factor):
    """Scale the image by the specified scale factor."""
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    dim = (width, height)
    scaled = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return scaled

def apply_brightness(image, brightness_level):
    """Adjust the brightness of the image."""
    brightness = brightness_level - 50  # Center at 0
    bright_img = cv2.convertScaleAbs(image, alpha=1, beta=brightness)
    return bright_img

def apply_augmentation_noise(image, noise_type='gaussian', noise_level=0.05):
    """Add noise to the image for augmentation purposes."""
    if noise_type == 'gaussian':
        row, col, ch = image.shape
        mean = 0
        sigma = noise_level * 255
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        noisy = image + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)
    elif noise_type == 'salt_pepper':
        s_vs_p = 0.5
        amount = noise_level
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]]
        for c in range(image.shape[2]):
            out[coords[0], coords[1], c] = 255

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]]
        for c in range(image.shape[2]):
            out[coords[0], coords[1], c] = 0
        return out
    return image

# ======================
# Augmentation Functions
# ======================

@st.cache_data
def apply_horizontal_flip(image):
    """Apply horizontal flip to the image."""
    transform = A.HorizontalFlip(p=1.0)
    augmented = transform(image=image)
    return augmented['image']

@st.cache_data
def apply_vertical_flip(image):
    """Apply vertical flip to the image."""
    transform = A.VerticalFlip(p=1.0)
    augmented = transform(image=image)
    return augmented['image']

@st.cache_data
def apply_rotate(image, angle=0):
    """Rotate the image by a specified angle."""
    transform = A.Rotate(limit=(angle, angle), p=1.0)
    augmented = transform(image=image)
    return augmented['image']

@st.cache_data
def apply_random_resized_crop(image, height=224, width=224, scale=(0.8, 1.0), ratio=(3./4., 4./3.)):
    """Apply random resized crop to the image."""
    transform = A.RandomResizedCrop(height=height, width=width, scale=scale, ratio=ratio, p=1.0)
    augmented = transform(image=image)
    return augmented['image']

@st.cache_data
def apply_color_jitter(image, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2):
    """Apply color jitter to the image."""
    transform = A.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue, p=1.0)
    augmented = transform(image=image)
    return augmented['image']

@st.cache_data
def apply_affine(image, shear=0, scale=1.0, translate_x=0, translate_y=0):
    """Apply affine transformation to the image."""
    transform = A.Affine(scale=scale, shear=shear, translate_percent={"x": translate_x, "y": translate_y}, p=1.0)
    augmented = transform(image=image)
    return augmented['image']

@st.cache_data
def apply_normalize(image):
    """Normalize the image using ImageNet statistics."""
    transform = A.Normalize(mean=(0.485, 0.456, 0.406), 
                           std=(0.229, 0.224, 0.225), 
                           max_pixel_value=255.0, p=1.0)
    augmented = transform(image=image)
    # Convert back to uint8 for display purposes
    augmented_image = (augmented['image'] * 255).astype(np.uint8)
    return augmented_image