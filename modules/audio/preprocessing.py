# modules/data_processing/image_preprocessing.py

import streamlit as st
import numpy as np
import cv2
from utils.helpers import (
    apply_noise,
    apply_salt_pepper_noise,
    apply_sobel_edge_detection,
    apply_laplacian_edge_detection,
    apply_resize,
    apply_grayscale,
    apply_blur,
    apply_edge_detection,
    apply_contrast_brightness,
    apply_color_space,
    apply_normalization
)
from PIL import Image
from io import BytesIO

def load_image_cv2(uploaded_file):
    """
    Load an uploaded image file using OpenCV.

    Parameters:
        uploaded_file (BytesIO): The uploaded image file.

    Returns:
        np.ndarray: The image in RGB format or None if loading fails.
    """
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Error loading image. Please upload a valid image file.")
        return None  # Return None on failure
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def image_preprocessing_tab():
    st.header("🎨 Image Preprocessing")
    st.markdown("""
    Enhance and prepare your images using various preprocessing techniques. The default preprocessings are applied automatically upon image upload. You can adjust parameters and reapply transformations as needed.
    """)

    # Initialize session state for images
    if 'original_pre_image' not in st.session_state:
        st.session_state.original_pre_image = None
    # Initialize preprocessed images
    preprocessed_keys = [
        'resize_image',
        'contrast_brightness_image',
        'blur_image',
        'noise_image',
        'edge_image',
        'color_space_image',
        'grayscale_image'  # Added Grayscale
    ]
    for key in preprocessed_keys:
        if key not in st.session_state:
            st.session_state[key] = None

    # File Uploader
    uploaded_image = st.file_uploader("📁 Upload Image for Preprocessing", type=['png', 'jpg', 'jpeg'])

    if uploaded_image is not None:
        image = load_image_cv2(uploaded_image)

        if image is not None:
            # Store original image in session state
            if st.session_state.original_pre_image is None:
                st.session_state.original_pre_image = image.copy()

            # Display Original Image
            st.subheader("📷 Original Image")
            st.image(st.session_state.original_pre_image, caption="Original Image", use_container_width=True)

            st.markdown("---")

            # Apply all preprocessing with default parameters
            if st.session_state.resize_image is None:
                st.session_state.resize_image = apply_resize(st.session_state.original_pre_image, width=224, height=224)
            if st.session_state.contrast_brightness_image is None:
                st.session_state.contrast_brightness_image = apply_contrast_brightness(st.session_state.original_pre_image, alpha=1.0, beta=0)
            if st.session_state.blur_image is None:
                st.session_state.blur_image = apply_blur(st.session_state.original_pre_image, blur_type='Gaussian', blur_kernel=5)
            if st.session_state.noise_image is None:
                st.session_state.noise_image = apply_noise(st.session_state.original_pre_image, noise_type='Gaussian', noise_level=0.05)
            if st.session_state.edge_image is None:
                st.session_state.edge_image = apply_edge_detection(st.session_state.original_pre_image, edge_algorithm='Canny')
            if st.session_state.color_space_image is None:
                st.session_state.color_space_image = apply_color_space(st.session_state.original_pre_image, color_space='RGB')
            if st.session_state.grayscale_image is None:
                st.session_state.grayscale_image = apply_grayscale(st.session_state.original_pre_image)

            # Display Preprocessed Images
            preprocessings = [
                {
                    'name': 'Resize',
                    'image_key': 'resize_image',
                    'controls': resize_controls,
                    'caption': '📏 Resized Image'
                },
                {
                    'name': 'Contrast & Brightness Adjustment',
                    'image_key': 'contrast_brightness_image',
                    'controls': contrast_brightness_controls,
                    'caption': '⚖️ Contrast & Brightness Adjusted Image'
                },
                {
                    'name': 'Blurring',
                    'image_key': 'blur_image',
                    'controls': blur_controls,
                    'caption': '🌀 Blurred Image'
                },
                {
                    'name': 'Noise Addition',
                    'image_key': 'noise_image',
                    'controls': noise_addition_controls,
                    'caption': '🧂 Noise Added Image'
                },
                {
                    'name': 'Edge Detection',
                    'image_key': 'edge_image',
                    'controls': edge_detection_controls,
                    'caption': '🔍 Edge Detected Image'
                },
                {
                    'name': 'Color Space Conversion',
                    'image_key': 'color_space_image',
                    'controls': color_space_controls,
                    'caption': '🎨 Color Space Image'
                },
                {
                    'name': 'Grayscale Conversion',
                    'image_key': 'grayscale_image',
                    'controls': grayscale_controls,  # Added Grayscale Controls
                    'caption': '🖤 Grayscale Image'
                }
            ]

            # Arrange preprocessings in a grid layout (2 per row)
            num_columns = 2
            for i in range(0, len(preprocessings), num_columns):
                cols = st.columns(num_columns)
                for j in range(num_columns):
                    if i + j < len(preprocessings):
                        preprocessing = preprocessings[i + j]
                        with cols[j]:
                            st.markdown(f"### {preprocessing['name']}")
                            preprocessing['controls']()
                            if st.session_state[preprocessing['image_key']] is not None:
                                if preprocessing['name'] == 'Color Space Conversion':
                                    display_color_space_image(preprocessing['image_key'], preprocessing['caption'])
                                else:
                                    st.image(
                                        st.session_state[preprocessing['image_key']],
                                        caption=preprocessing['caption'],
                                        use_container_width=True
                                    )
                                    # Download Button aligned with image
                                    download_image_button(preprocessing['image_key'], preprocessing['name'], preprocessing['caption'])

            st.markdown("---")

def display_color_space_image(state_key, caption):
    """Display color space converted image using Streamlit's st.image."""
    img = st.session_state[state_key]
    if img is not None:
        st.image(img, caption=caption, use_container_width=True)
        # Download Button
        download_image_button(state_key, 'Color Space', caption)

def download_image_button(state_key, technique_name, caption):
    """Provide a download button for the processed image."""
    img = st.session_state[state_key]
    if img is not None:
        # Convert NumPy array to PIL Image
        img_pil = Image.fromarray(img)
        # Allow user to select format
        file_format = st.selectbox(
            f"Select format for {technique_name} Image",
            options=['PNG', 'JPEG'],
            key=f"download_format_{technique_name}"
        )
        # Save image to BytesIO
        buf = BytesIO()
        if file_format == 'PNG':
            img_pil.save(buf, format='PNG')
            mime_type = 'image/png'
            file_ext = 'png'
        else:
            img_pil.save(buf, format='JPEG')
            mime_type = 'image/jpeg'
            file_ext = 'jpg'
        byte_im = buf.getvalue()
        st.download_button(
            label=f"📥 Download {technique_name} Image",
            data=byte_im,
            file_name=f"{technique_name.lower().replace(' ', '_')}_image.{file_ext}",
            mime=mime_type
        )

# Controls for each selected preprocessing technique

def resize_controls():
    """Controls for Resize preprocessing."""
    col1, col2 = st.columns(2)
    with col1:
        resize_width = st.number_input(
            "Width (pixels)",
            min_value=32,
            max_value=4096,
            value=224,
            step=1,
            key="resize_width",
            help="Enter the desired width for resizing the image."
        )
    with col2:
        resize_height = st.number_input(
            "Height (pixels)",
            min_value=32,
            max_value=4096,
            value=224,
            step=1,
            key="resize_height",
            help="Enter the desired height for resizing the image."
        )
    if st.button("Reapply Resize", key="reapply_resize_btn"):
        try:
            st.session_state.resize_image = apply_resize(
                st.session_state.original_pre_image, resize_width, resize_height
            )
            # Success message removed
        except Exception as e:
            st.error(f"Error resizing image: {e}")

def contrast_brightness_controls():
    """Controls for Contrast & Brightness Adjustment."""
    col1, col2 = st.columns(2)
    with col1:
        contrast_alpha = st.slider(
            "Contrast (α)",
            0.0,
            3.0,
            1.0,
            0.1,
            key="contrast_alpha_slider",
            help="Adjust the contrast of the image. α > 1 increases contrast."
        )
    with col2:
        brightness_beta = st.slider(
            "Brightness (β)",
            -100,
            100,
            0,
            1,
            key="brightness_beta_slider",
            help="Adjust the brightness of the image. β > 0 increases brightness."
        )
    if st.button("Reapply Contrast & Brightness", key="reapply_contrast_brightness_btn"):
        try:
            st.session_state.contrast_brightness_image = apply_contrast_brightness(
                st.session_state.original_pre_image, contrast_alpha, brightness_beta
            )
            # Success message removed
        except Exception as e:
            st.error(f"Error adjusting Contrast & Brightness: {e}")

def blur_controls():
    """Controls for Blurring."""
    blur_type = st.selectbox(
        "Select Blur Type",
        ['None', 'Gaussian', 'Median', 'Bilateral'],
        key="blur_type_select_box",
        help="Choose the type of blurring to apply."
    )
    if blur_type != 'None':
        blur_kernel = st.slider(
            "Blur Kernel Size (Odd)",
            1,
            31,
            3,
            step=2,
            key="blur_kernel_size",
            help="Select the kernel size for blurring. Must be an odd number."
        )
        if st.button(f"Reapply {blur_type} Blur", key="reapply_blur_btn"):
            try:
                st.session_state.blur_image = apply_blur(
                    st.session_state.original_pre_image, blur_type, blur_kernel
                )
                # Success message removed
            except Exception as e:
                st.error(f"Error applying {blur_type} Blur: {e}")

def noise_addition_controls():
    """Controls for Noise Addition."""
    noise_type = st.selectbox(
        "Select Noise Type",
        ['None', 'Gaussian', 'Salt & Pepper'],
        key="noise_type_select_box",
        help="Choose the type of noise to add to the image."
    )
    if noise_type != 'None':
        noise_level = st.slider(
            "Noise Level",
            0.0,
            0.1,
            0.05,
            0.01,
            key="noise_level_slider",
            help="Adjust the intensity of the noise to be added."
        )
        if st.button(f"Reapply {noise_type} Noise", key="reapply_noise_btn"):
            try:
                if noise_type == 'Gaussian':
                    st.session_state.noise_image = apply_noise(
                        st.session_state.original_pre_image, noise_type='gaussian', noise_level=noise_level
                    )
                elif noise_type == 'Salt & Pepper':
                    st.session_state.noise_image = apply_salt_pepper_noise(
                        st.session_state.original_pre_image, noise_level=noise_level
                    )
                # Success message removed
            except Exception as e:
                st.error(f"Error applying {noise_type} Noise: {e}")

def edge_detection_controls():
    """Controls for Edge Detection."""
    edge_algorithm = st.selectbox(
        "Select Edge Detection Algorithm",
        ['None', 'Canny', 'Sobel', 'Laplacian'],
        key="edge_algorithm_select_box",
        help="Choose the edge detection algorithm to apply."
    )
    if edge_algorithm != 'None':
        if st.button(f"Reapply {edge_algorithm} Edge Detection", key="reapply_edge_btn"):
            try:
                st.session_state.edge_image = apply_edge_detection(
                    st.session_state.original_pre_image, edge_algorithm
                )
                # Success message removed
            except Exception as e:
                st.error(f"Error applying {edge_algorithm} Edge Detection: {e}")

def color_space_controls():
    """Controls for Color Space Conversion."""
    color_space = st.selectbox(
        "Select Color Space",
        ['RGB', 'HSV', 'LAB', 'YCrCb'],
        key="color_space_select_box",
        help="Choose the color space to convert the image to."
    )
    if color_space != 'RGB':
        if st.button(f"Reapply {color_space} Color Space Conversion", key="reapply_color_space_btn"):
            try:
                st.session_state.color_space_image = apply_color_space(
                    st.session_state.original_pre_image, color_space
                )
                # Success message removed
            except Exception as e:
                st.error(f"Error converting to {color_space} Color Space: {e}")

def grayscale_controls():
    """Controls for Grayscale Conversion."""
    if st.button("Reapply Grayscale", key="reapply_grayscale_btn"):
        try:
            st.session_state.grayscale_image = apply_grayscale(st.session_state.original_pre_image)
            # Success message removed
        except Exception as e:
            st.error(f"Error applying Grayscale Conversion: {e}")
