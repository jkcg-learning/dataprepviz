import streamlit as st
import numpy as np
from PIL import Image
import cv2
import torch
import torchvision.transforms as transforms
from utils.helpers import apply_noise

def image_preprocessing_tab(method):
    st.header(f"Image Preprocessing - {method} Methods")
    col1, col2 = st.columns(2)

    with col1:
        uploaded_image = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])

        if uploaded_image is not None:
            if method == "Classical":
                image = load_image_cv2(uploaded_image)
                st.image(image, caption="Original Image", use_column_width=True)

                # Classical preprocessing options
                classical_preprocessing(image, col2)
            else:
                image = load_image_pil(uploaded_image)
                st.image(image, caption="Original Image", use_column_width=True)

                # Deep Learning preprocessing options
                deeplearning_preprocessing(image, col2)

def load_image_cv2(uploaded_file):
    """Load an uploaded image file using OpenCV."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def load_image_pil(uploaded_file):
    """Load an uploaded image file using PIL."""
    image = Image.open(uploaded_file).convert('RGB')
    return image

def classical_preprocessing(image, col2):
    # Preprocessing options
    st.subheader("Preprocessing Options")

    # Basic options
    resize_dim = st.slider("Resize Dimension", 32, 512, 224)
    normalize = st.checkbox("Normalize", value=True)
    grayscale = st.checkbox("Convert to Grayscale")

    # Advanced options
    blur_type = st.selectbox("Blur Type", ['None', 'Gaussian', 'Median', 'Bilateral'])
    if blur_type != 'None':
        blur_kernel = st.slider("Blur Kernel Size", 1, 15, 3, step=2)

    edge_detection = st.checkbox("Edge Detection")
    contrast_alpha = st.slider("Contrast", 0.0, 3.0, 1.0)
    brightness_beta = st.slider("Brightness", -100, 100, 0)

    color_space = st.selectbox("Color Space", ['RGB', 'HSV', 'LAB', 'YCrCb'])

    if st.button("Preprocess Image"):
        processed_img = image.copy()

        # Resize
        processed_img = cv2.resize(processed_img, (resize_dim, resize_dim))

        # Color space conversion
        if color_space != 'RGB':
            color_conversion_codes = {
                'HSV': cv2.COLOR_RGB2HSV,
                'LAB': cv2.COLOR_RGB2LAB,
                'YCrCb': cv2.COLOR_RGB2YCrCb
            }
            processed_img = cv2.cvtColor(processed_img, color_conversion_codes[color_space])

        # Grayscale
        if grayscale:
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2GRAY)
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)

        # Blur
        if blur_type != 'None':
            if blur_type == 'Gaussian':
                processed_img = cv2.GaussianBlur(processed_img, (blur_kernel, blur_kernel), 0)
            elif blur_type == 'Median':
                processed_img = cv2.medianBlur(processed_img, blur_kernel)
            elif blur_type == 'Bilateral':
                processed_img = cv2.bilateralFilter(processed_img, blur_kernel, 75, 75)

        # Edge detection
        if edge_detection:
            gray = cv2.cvtColor(processed_img, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            processed_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        # Contrast and brightness
        processed_img = cv2.convertScaleAbs(processed_img, alpha=contrast_alpha, beta=brightness_beta)

        # Normalize
        if normalize:
            processed_img = cv2.normalize(processed_img, None, 0, 255, cv2.NORM_MINMAX)

        with col2:
            st.subheader("Preprocessed Result")
            st.image(processed_img, caption="Preprocessed Image", use_column_width=True)

def deeplearning_preprocessing(image, col2):
    st.subheader("Preprocessing Options")

    resize_dim = st.slider("Resize Dimension", 32, 512, 128)
    normalize = st.checkbox("Normalize", value=True)

    if st.button("Preprocess Image"):
        preprocess_transforms = [transforms.Resize((resize_dim, resize_dim)), transforms.ToTensor()]

        if normalize:
            preprocess_transforms.append(
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            )

        transform = transforms.Compose(preprocess_transforms)
        image_tensor = transform(image)
        processed_img = image_tensor.permute(1, 2, 0).numpy()
        processed_img = (processed_img * 0.5 + 0.5) * 255  # Denormalize for display

        with col2:
            st.subheader("Preprocessed Result")
            st.image(processed_img.astype(np.uint8), caption="Preprocessed Image", use_column_width=True)
