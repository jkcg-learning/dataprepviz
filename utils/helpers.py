import numpy as np
import cv2

def apply_noise(image, noise_type='gaussian'):
    """Apply noise to an image."""
    if noise_type == 'gaussian':
        row, col, ch = image.shape
        mean = 0
        sigma = 25
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        noisy = image + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)
    return image

