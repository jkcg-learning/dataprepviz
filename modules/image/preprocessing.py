# modules/data_processing/image_preprocessing.py

import streamlit as st
import numpy as np
import cv2
from utils.helpers import (
    apply_noise,
    apply_salt_pepper_noise,
    apply_sobel_edge_detection,
    apply_laplacian_edge_detection
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

def apply_resize(image, width, height):
    """Resize the image to the specified width and height."""
    resized = cv2.resize(image, (width, height))
    return resized

def apply_grayscale(image):
    """Convert the image to grayscale."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return gray_rgb

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

def apply_contrast_brightness(image, alpha, beta):
    """Adjust the contrast and brightness of the image."""
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

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

def apply_normalization(image):
    """Normalize the image pixel values to the range [0, 255]."""
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return normalized

def preprocess_image_all(
    image, resize_dim, resize_width, resize_height, normalize, grayscale, blur_type, blur_kernel,
    edge_algorithm, contrast_alpha, brightness_beta, color_space, noise_type, noise_level
):
    """Apply all selected preprocessing steps to the image."""
    img = image.copy()

    # Apply Resize
    if resize_dim:
        img = apply_resize(img, resize_width, resize_height)

    # Apply Color Space Conversion
    if color_space != 'RGB':
        img = apply_color_space(img, color_space)

    # Apply Grayscale Conversion
    if grayscale:
        img = apply_grayscale(img)

    # Apply Blurring
    if blur_type != 'None':
        img = apply_blur(img, blur_type, blur_kernel)

    # Apply Edge Detection
    if edge_algorithm != 'None':
        img = apply_edge_detection(img, edge_algorithm)

    # Apply Contrast and Brightness Adjustment
    if contrast_alpha != 1.0 or brightness_beta != 0:
        img = apply_contrast_brightness(img, contrast_alpha, brightness_beta)

    # Apply Normalization
    if normalize:
        img = apply_normalization(img)

    # Apply Noise Addition
    if noise_type != 'None':
        if noise_type == 'Gaussian':
            img = apply_noise(img, noise_type='gaussian', noise_level=noise_level)
        elif noise_type == 'Salt & Pepper':
            img = apply_salt_pepper_noise(img, noise_level=noise_level)

    return img

def image_preprocessing_tab():
    st.header("🎨 Image Preprocessing")
    st.markdown("""
    Enhance and prepare your images using various preprocessing techniques.
    """)

    # Initialize session state for images
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    # Initialize individual preprocessed images
    preprocessed_keys = [
        'resize_image', 'contrast_brightness_image',
        'blur_image', 'noise_image',
        'edge_image', 'color_space_image',
        'grayscale_image'  # Added Grayscale
    ]
    for key in preprocessed_keys:
        if key not in st.session_state:
            st.session_state[key] = None

    # File Uploader
    uploaded_image = st.file_uploader("📁 Upload Image", type=['png', 'jpg', 'jpeg'])

    if uploaded_image is not None:
        image = load_image_cv2(uploaded_image)

        if image is not None:
            # Store original image in session state
            if st.session_state.original_image is None:
                st.session_state.original_image = image.copy()

            # Display Original Image
            st.subheader("📷 Original Image")
            st.image(st.session_state.original_image, caption="Original Image", use_container_width=True)

            st.markdown("---")

            # Define Preprocessing Techniques in Desired Order
            techniques = [
                {
                    'name': 'Resize',
                    'controls': resize_controls,
                    'result_key': 'resize_image',
                    'caption': '🔄 Resized Image'
                },
                {
                    'name': 'Contrast & Brightness Adjustment',
                    'controls': contrast_brightness_controls,
                    'result_key': 'contrast_brightness_image',
                    'caption': '⚖️ Contrast & Brightness Adjusted Image'
                },
                {
                    'name': 'Blurring',
                    'controls': blur_controls,
                    'result_key': 'blur_image',
                    'caption': '🌀 Blurred Image'
                },
                {
                    'name': 'Noise Addition',
                    'controls': noise_addition_controls,
                    'result_key': 'noise_image',
                    'caption': '🧂 Noise Added Image'
                },
                {
                    'name': 'Edge Detection',
                    'controls': edge_detection_controls,
                    'result_key': 'edge_image',
                    'caption': '🔍 Edge Detected Image'
                },
                {
                    'name': 'Color Space Conversion',
                    'controls': color_space_controls,
                    'result_key': 'color_space_image',
                    'caption': '🎨 Color Space Image'
                },
                {
                    'name': 'Grayscale Conversion',
                    'controls': grayscale_controls,  # Added Grayscale Controls
                    'result_key': 'grayscale_image',
                    'caption': '🖤 Grayscale Image'
                }
            ]

            # Arrange techniques in a grid layout (2 per row)
            num_columns = 2
            for i in range(0, len(techniques), num_columns):
                cols = st.columns(num_columns)
                for j in range(num_columns):
                    if i + j < len(techniques):
                        technique = techniques[i + j]
                        with cols[j]:
                            st.markdown(f"### {technique['name']}")
                            technique['controls']()
                            if st.session_state[technique['result_key']] is not None:
                                if technique['name'] == 'Color Space Conversion':
                                    display_color_space_image(technique['result_key'], technique['caption'])
                                else:
                                    st.image(
                                        st.session_state[technique['result_key']],
                                        caption=technique['caption'],
                                        use_container_width=True
                                    )
                                    # Download Button aligned with image
                                    download_image_button(technique['result_key'], technique['name'], technique['caption'])

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
        # Save image to BytesIO
        buf = BytesIO()
        img_pil.save(buf, format='PNG')
        byte_im = buf.getvalue()
        st.download_button(
            label=f"📥 Download {technique_name} Image",
            data=byte_im,
            file_name=f"{technique_name.lower().replace(' ', '_')}_image.png",
            mime='image/png'
        )

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
    if st.button("Apply Resize", key="apply_resize_btn"):
        try:
            st.session_state.resize_image = apply_resize(
                st.session_state.original_image, resize_width, resize_height
            )
            # st.success(f"✅ Image resized to {resize_width}x{resize_height} pixels.")  # Removed
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
    if st.button("Apply Contrast & Brightness", key="apply_contrast_brightness_btn"):
        try:
            st.session_state.contrast_brightness_image = apply_contrast_brightness(
                st.session_state.original_image, contrast_alpha, brightness_beta
            )
            # st.success(f"✅ Contrast (α={contrast_alpha}) and Brightness (β={brightness_beta}) adjusted.")  # Removed
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
        if st.button(f"Apply {blur_type} Blur", key="apply_blur_btn"):
            try:
                st.session_state.blur_image = apply_blur(
                    st.session_state.original_image, blur_type, blur_kernel
                )
                # st.success(f"✅ {blur_type} Blur applied with kernel size {blur_kernel}.")  # Removed
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
        if st.button(f"Apply {noise_type} Noise", key="apply_noise_btn"):
            try:
                if noise_type == 'Gaussian':
                    st.session_state.noise_image = apply_noise(
                        st.session_state.original_image, noise_type='gaussian', noise_level=noise_level
                    )
                    # st.success(f"✅ Gaussian Noise applied with level {noise_level}.")  # Removed
                elif noise_type == 'Salt & Pepper':
                    st.session_state.noise_image = apply_salt_pepper_noise(
                        st.session_state.original_image, noise_level=noise_level
                    )
                    # st.success(f"✅ Salt & Pepper Noise applied with level {noise_level}.")  # Removed
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
        if st.button(f"Apply {edge_algorithm} Edge Detection", key="apply_edge_btn"):
            try:
                st.session_state.edge_image = apply_edge_detection(
                    st.session_state.original_image, edge_algorithm
                )
                # st.success(f"✅ {edge_algorithm} Edge Detection applied.")  # Removed
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
        if st.button(f"Apply {color_space} Color Space Conversion", key="apply_color_space_btn"):
            try:
                st.session_state.color_space_image = apply_color_space(
                    st.session_state.original_image, color_space
                )
                # st.success(f"✅ Color Space converted to {color_space}.")  # Removed
            except Exception as e:
                st.error(f"Error converting to {color_space} Color Space: {e}")

def grayscale_controls():
    """Controls for Grayscale Conversion."""
    if st.button("Apply Grayscale", key="apply_grayscale_btn"):
        try:
            st.session_state.grayscale_image = apply_grayscale(st.session_state.original_image)
            # st.success("✅ Grayscale Conversion applied.")  # Removed
        except Exception as e:
            st.error(f"Error applying Grayscale Conversion: {e}")
