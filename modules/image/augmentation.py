import streamlit as st
import numpy as np
from PIL import Image
import cv2
import torch
import torchvision.transforms as transforms
from utils.helpers import apply_noise

def image_augmentation_tab(method):
    st.header(f"Image Augmentation - {method} Methods")
    col1, col2 = st.columns(2)

    with col1:
        uploaded_image = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])

        if uploaded_image is not None:
            if method == "Classical":
                image = load_image_cv2(uploaded_image)
                st.image(image, caption="Original Image", use_column_width=True)

                # Classical augmentation options
                classical_augmentation(image, col2)
            else:
                image = load_image_pil(uploaded_image)
                st.image(image, caption="Original Image", use_column_width=True)

                # Deep Learning augmentation options
                deeplearning_augmentation(image, col2)

def load_image_cv2(uploaded_file):
    """Load an uploaded image file using OpenCV."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def load_image_pil(uploaded_file):
    """Load an uploaded image file using PIL."""
    image = Image.open(uploaded_file).convert('RGB')
    return image

def classical_augmentation(image, col2):
    # Augmentation options
    st.subheader("Augmentation Options")
    aug_col1, aug_col2 = st.columns(2)

    with aug_col1:
        rotation_angle = st.slider("Rotation Angle", -180, 180, 0)
        scale_factor = st.slider("Scale Factor", 0.5, 2.0, 1.0)
        add_noise = st.checkbox("Add Noise")

    with aug_col2:
        flip_h = st.checkbox("Horizontal Flip")
        flip_v = st.checkbox("Vertical Flip")
        perspective_transform = st.checkbox("Perspective Transform")

    if st.button("Apply Augmentations"):
        augmented_images = []
        captions = []

        # Flip augmentations
        if flip_h:
            flipped_h = cv2.flip(image, 1)
            augmented_images.append(flipped_h)
            captions.append("Horizontal Flip")

        if flip_v:
            flipped_v = cv2.flip(image, 0)
            augmented_images.append(flipped_v)
            captions.append("Vertical Flip")

        # Rotation and scaling
        if rotation_angle != 0 or scale_factor != 1.0:
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, scale_factor)
            rotated_scaled = cv2.warpAffine(image, rotation_matrix, (width, height))
            augmented_images.append(rotated_scaled)
            captions.append(f"Rotation {rotation_angle}° Scale {scale_factor:.1f}")

        # Noise
        if add_noise:
            noisy_img = apply_noise(image)
            augmented_images.append(noisy_img)
            captions.append("Gaussian Noise")

        # Perspective transform
        if perspective_transform:
            height, width = image.shape[:2]
            src_points = np.float32([[0, 0], [width-1, 0], [0, height-1], [width-1, height-1]])
            dst_points = np.float32([
                [width*0.1, height*0.1],
                [width*0.9, height*0.1],
                [width*0.1, height*0.9],
                [width*0.9, height*0.9]
            ])
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            perspective = cv2.warpPerspective(image, matrix, (width, height))
            augmented_images.append(perspective)
            captions.append("Perspective Transform")

        with col2:
            st.subheader("Augmentation Results")
            for img, caption in zip(augmented_images, captions):
                st.image(img, caption=caption, use_column_width=True)

def deeplearning_augmentation(image, col2):
    st.subheader("Augmentation Options")

    aug_transforms = []

    if st.checkbox("Random Horizontal Flip"):
        aug_transforms.append(transforms.RandomHorizontalFlip())

    if st.checkbox("Random Rotation"):
        rotation_degrees = st.slider("Rotation Degrees", 0, 360, 10)
        aug_transforms.append(transforms.RandomRotation(rotation_degrees))

    if st.checkbox("Color Jitter"):
        brightness = st.slider("Brightness", 0.0, 1.0, 0.2)
        contrast = st.slider("Contrast", 0.0, 1.0, 0.2)
        aug_transforms.append(transforms.ColorJitter(brightness=brightness, contrast=contrast))

    if st.button("Apply Augmentations"):
        transform = transforms.Compose(aug_transforms + [transforms.ToTensor()])
        augmented_images = []
        captions = []

        for i in range(3):  # Generate 3 augmented images
            augmented_img_tensor = transform(image)
            augmented_img = augmented_img_tensor.permute(1, 2, 0).numpy()
            augmented_img = (augmented_img * 0.5 + 0.5) * 255  # Denormalize
            augmented_images.append(augmented_img.astype(np.uint8))
            captions.append(f"Augmented Image {i+1}")

        with col2:
            st.subheader("Augmentation Results")
            for img, caption in zip(augmented_images, captions):
                st.image(img, caption=caption, use_column_width=True)
