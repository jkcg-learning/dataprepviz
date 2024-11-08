import numpy as np
import cv2

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
