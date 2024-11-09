# utils/helpers.py

import numpy as np
import cv2
import albumentations as A
import streamlit as st

# ======================
# Preprocessing Functions
# ======================

@st.cache_data
def apply_resize(image, width, height):
    """Resize the image to the specified width and height."""
    resized = cv2.resize(image, (width, height))
    return resized

@st.cache_data
def apply_grayscale(image):
    """Convert the image to grayscale."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return gray_rgb

@st.cache_data
def apply_blur(image, blur_type, blur_kernel):
    """Apply the specified blur to the image."""
    if blur_type == 'Gaussian':
        blurred = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
    elif blur_type == 'Median':
        blurred = cv2.medianBlur(image, blur_kernel)
    elif blur_type == 'Bilateral':
        blurred = cv2.bilateralFilter(image, blur_kernel, 75, 75)
    else:
        blurred = image
    return blurred

@st.cache_data
def apply_edge_detection(image, edge_algorithm):
    """Apply the specified edge detection algorithm to the image."""
    if edge_algorithm == 'Canny':
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return edges_rgb
    elif edge_algorithm == 'Sobel':
        return apply_sobel_edge_detection(image)
    elif edge_algorithm == 'Laplacian':
        return apply_laplacian_edge_detection(image)
    else:
        return image

@st.cache_data
def apply_contrast_brightness(image, alpha, beta):
    """Adjust the contrast and brightness of the image."""
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

@st.cache_data
def apply_color_space(image, color_space):
    """Convert the image to the specified color space and back to RGB for display."""
    color_spaces = {
        'RGB': image,
        'HSV': cv2.cvtColor(image, cv2.COLOR_RGB2HSV),
        'LAB': cv2.cvtColor(image, cv2.COLOR_RGB2LAB),
        'YCrCb': cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    }
    if color_space in color_spaces:
        converted = color_spaces[color_space]
        return converted
    return image

@st.cache_data
def apply_normalization(image):
    """Normalize the image pixel values to the range [0, 255]."""
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return normalized

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

# ======================
# Additional Helper Functions
# ======================

@st.cache_data
def apply_noise(image, noise_type='gaussian', noise_level=0.05):
    """Apply specified noise to the image."""
    if noise_type == 'gaussian':
        row, col, ch = image.shape
        mean = 0
        sigma = noise_level * 255
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        noisy = image + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)
    elif noise_type == 'salt_pepper':
        return apply_salt_pepper_noise(image, noise_level=noise_level)
    return image

@st.cache_data
def apply_salt_pepper_noise(image, noise_level=0.05):
    """Apply Salt & Pepper noise to the image."""
    s_vs_p = 0.5
    amount = noise_level
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]]
    out[coords[0], coords[1], :] = 255
    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]]
    out[coords[0], coords[1], :] = 0
    return out

@st.cache_data
def apply_sobel_edge_detection(image):
    """Apply Sobel edge detection to the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = np.uint8(np.clip(sobel, 0, 255))
    sobel_rgb = cv2.cvtColor(sobel, cv2.COLOR_GRAY2RGB)
    return sobel_rgb

@st.cache_data
def apply_laplacian_edge_detection(image):
    """Apply Laplacian edge detection to the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    laplacian_rgb = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2RGB)
    return laplacian_rgb
