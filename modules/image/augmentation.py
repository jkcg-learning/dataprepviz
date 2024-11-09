# modules/data_processing/image_augmentation.py

import streamlit as st
import numpy as np
import cv2
from utils.helpers import (
    apply_horizontal_flip,
    apply_vertical_flip,
    apply_rotate,
    apply_random_resized_crop,
    apply_color_jitter,
    apply_affine,
    apply_normalize
)
from PIL import Image
from io import BytesIO

def image_augmentation_tab():
    st.header("🛠️ Image Augmentation")
    st.markdown("""
    Automatically apply default augmentation techniques upon image upload. You can adjust parameters and reapply transformations as needed.
    """)

    # Initialize session state for images
    if 'original_aug_image' not in st.session_state:
        st.session_state.original_aug_image = None
    # Initialize augmented images
    augmented_keys = [
        'horizontal_flip_image',
        'vertical_flip_image',
        'rotate_image',
        'random_resized_crop_image',
        'color_jitter_image',
        'affine_image',
        'normalize_image'
    ]
    for key in augmented_keys:
        if key not in st.session_state:
            st.session_state[key] = None

    # File Uploader
    uploaded_image = st.file_uploader("📁 Upload Image for Augmentation", type=['png', 'jpg', 'jpeg'])

    if uploaded_image is not None:
        image = load_image_cv2(uploaded_image)

        if image is not None:
            # Store original image in session state
            if st.session_state.original_aug_image is None:
                st.session_state.original_aug_image = image.copy()

            # Display Original Image
            st.subheader("📷 Original Image")
            st.image(st.session_state.original_aug_image, caption="Original Image", use_container_width=True)

            st.markdown("---")

            # Apply all augmentations with default parameters
            if st.session_state.horizontal_flip_image is None:
                st.session_state.horizontal_flip_image = apply_horizontal_flip(st.session_state.original_aug_image)
            if st.session_state.vertical_flip_image is None:
                st.session_state.vertical_flip_image = apply_vertical_flip(st.session_state.original_aug_image)
            if st.session_state.rotate_image is None:
                st.session_state.rotate_image = apply_rotate(st.session_state.original_aug_image, angle=0)
            if st.session_state.random_resized_crop_image is None:
                st.session_state.random_resized_crop_image = apply_random_resized_crop(st.session_state.original_aug_image)
            if st.session_state.color_jitter_image is None:
                st.session_state.color_jitter_image = apply_color_jitter(st.session_state.original_aug_image)
            if st.session_state.affine_image is None:
                st.session_state.affine_image = apply_affine(st.session_state.original_aug_image)
            if st.session_state.normalize_image is None:
                st.session_state.normalize_image = apply_normalize(st.session_state.original_aug_image)

            # Display Augmented Images
            augmentations = [
                {
                    'name': 'Horizontal Flip',
                    'image_key': 'horizontal_flip_image',
                    'controls': horizontal_flip_controls,
                    'caption': '🔄 Horizontally Flipped Image'
                },
                {
                    'name': 'Vertical Flip',
                    'image_key': 'vertical_flip_image',
                    'controls': vertical_flip_controls,
                    'caption': '🔁 Vertically Flipped Image'
                },
                {
                    'name': 'Rotate',
                    'image_key': 'rotate_image',
                    'controls': rotate_controls,
                    'caption': '🔄 Rotated Image'
                },
                {
                    'name': 'Random Resized Crop',
                    'image_key': 'random_resized_crop_image',
                    'controls': random_resized_crop_controls,
                    'caption': '✂️ Randomly Resized Cropped Image'
                },
                {
                    'name': 'Color Jitter',
                    'image_key': 'color_jitter_image',
                    'controls': color_jitter_controls,
                    'caption': '🌈 Color Jittered Image'
                },
                {
                    'name': 'Affine Transformation',
                    'image_key': 'affine_image',
                    'controls': affine_controls,
                    'caption': '🔀 Affine Transformed Image'
                },
                {
                    'name': 'Normalize',
                    'image_key': 'normalize_image',
                    'controls': normalize_controls,
                    'caption': '📊 Normalized Image'
                }
            ]

            # Arrange augmentations in a grid layout (2 per row)
            num_columns = 2
            for i in range(0, len(augmentations), num_columns):
                cols = st.columns(num_columns)
                for j in range(num_columns):
                    if i + j < len(augmentations):
                        augmentation = augmentations[i + j]
                        with cols[j]:
                            st.markdown(f"### {augmentation['name']}")
                            augmentation['controls'](augmentation['image_key'])
                            if st.session_state[augmentation['image_key']] is not None:
                                st.image(
                                    st.session_state[augmentation['image_key']],
                                    caption=augmentation['caption'],
                                    use_container_width=True
                                )
                                # Download Button aligned with image
                                download_image_button(augmentation['image_key'], augmentation['name'], augmentation['caption'])

            st.markdown("---")

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

def download_image_button(state_key, technique_name, caption):
    """Provide a download button for the augmented image."""
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

# Controls for each selected augmentation technique

def horizontal_flip_controls(image_key):
    """Controls for Horizontal Flip."""
    if st.button("Reapply Horizontal Flip", key=f"reapply_horizontal_flip_btn_{image_key}"):
        try:
            st.session_state[image_key] = apply_horizontal_flip(st.session_state.original_aug_image)
            # Success message removed
        except Exception as e:
            st.error(f"Error applying Horizontal Flip: {e}")

def vertical_flip_controls(image_key):
    """Controls for Vertical Flip."""
    if st.button("Reapply Vertical Flip", key=f"reapply_vertical_flip_btn_{image_key}"):
        try:
            st.session_state[image_key] = apply_vertical_flip(st.session_state.original_aug_image)
            # Success message removed
        except Exception as e:
            st.error(f"Error applying Vertical Flip: {e}")

def rotate_controls(image_key):
    """Controls for Rotate."""
    angle = st.slider(
        "Rotation Angle (degrees)",
        -180.0,
        180.0,
        0.0,
        1.0,
        key=f"{image_key}_angle_slider",
        help="Rotate the image by the specified angle."
    )
    if st.button("Reapply Rotate", key=f"reapply_rotate_btn_{image_key}"):
        try:
            st.session_state[image_key] = apply_rotate(st.session_state.original_aug_image, angle)
            # Success message removed
        except Exception as e:
            st.error(f"Error applying Rotate: {e}")

def random_resized_crop_controls(image_key):
    """Controls for Random Resized Crop."""
    crop_height = st.number_input(
        "Crop Height (pixels)",
        min_value=10,
        max_value=st.session_state.original_aug_image.shape[0],
        value=224,
        step=1,
        key=f"{image_key}_crop_height",
        help="Height of the cropped area."
    )
    crop_width = st.number_input(
        "Crop Width (pixels)",
        min_value=10,
        max_value=st.session_state.original_aug_image.shape[1],
        value=224,
        step=1,
        key=f"{image_key}_crop_width",
        help="Width of the cropped area."
    )
    scale_min = st.slider(
        "Scale Min",
        0.1,
        1.0,
        0.8,
        0.05,
        key=f"{image_key}_scale_min",
        help="Minimum scale for the resized crop."
    )
    scale_max = st.slider(
        "Scale Max",
        0.1,
        1.0,
        1.0,
        0.05,
        key=f"{image_key}_scale_max",
        help="Maximum scale for the resized crop."
    )
    aspect_ratio_min = st.slider(
        "Aspect Ratio Min",
        0.5,
        2.0,
        0.75,
        0.05,
        key=f"{image_key}_aspect_ratio_min",
        help="Minimum aspect ratio for the resized crop."
    )
    aspect_ratio_max = st.slider(
        "Aspect Ratio Max",
        0.5,
        2.0,
        1.33,
        0.05,
        key=f"{image_key}_aspect_ratio_max",
        help="Maximum aspect ratio for the resized crop."
    )
    if st.button("Reapply Random Resized Crop", key=f"reapply_random_resized_crop_btn_{image_key}"):
        try:
            st.session_state[image_key] = apply_random_resized_crop(
                st.session_state.original_aug_image,
                height=crop_height,
                width=crop_width,
                scale=(scale_min, scale_max),
                ratio=(aspect_ratio_min, aspect_ratio_max)
            )
            # Success message removed
        except Exception as e:
            st.error(f"Error applying Random Resized Crop: {e}")

def color_jitter_controls(image_key):
    """Controls for Color Jitter."""
    brightness = st.slider(
        "Brightness",
        0.0,
        0.5,
        0.2,
        0.05,
        key=f"{image_key}_brightness_slider",
        help="Adjust the brightness of the image."
    )
    contrast = st.slider(
        "Contrast",
        0.0,
        0.5,
        0.2,
        0.05,
        key=f"{image_key}_contrast_slider",
        help="Adjust the contrast of the image."
    )
    saturation = st.slider(
        "Saturation",
        0.0,
        0.5,
        0.2,
        0.05,
        key=f"{image_key}_saturation_slider",
        help="Adjust the saturation of the image."
    )
    hue = st.slider(
        "Hue",
        -0.5,
        0.5,
        0.2,
        0.05,
        key=f"{image_key}_hue_slider",
        help="Adjust the hue of the image."
    )
    if st.button("Reapply Color Jitter", key=f"reapply_color_jitter_btn_{image_key}"):
        try:
            st.session_state[image_key] = apply_color_jitter(
                st.session_state.original_aug_image,
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue
            )
            # Success message removed
        except Exception as e:
            st.error(f"Error applying Color Jitter: {e}")

def affine_controls(image_key):
    """Controls for Affine Transformation."""
    shear = st.slider(
        "Shear Angle (degrees)",
        -45.0,
        45.0,
        0.0,
        1.0,
        key=f"{image_key}_shear_slider",
        help="Apply shear to the image by the specified angle."
    )
    scale = st.slider(
        "Scale Factor",
        0.5,
        2.0,
        1.0,
        0.1,
        key=f"{image_key}_scale_slider",
        help="Scale the image by the specified factor."
    )
    translate_x = st.slider(
        "Translate X (% of image width)",
        -0.5,
        0.5,
        0.0,
        0.01,
        key=f"{image_key}_translate_x_slider",
        help="Translate the image along the X-axis."
    )
    translate_y = st.slider(
        "Translate Y (% of image height)",
        -0.5,
        0.5,
        0.0,
        0.01,
        key=f"{image_key}_translate_y_slider",
        help="Translate the image along the Y-axis."
    )
    if st.button("Reapply Affine Transformation", key=f"reapply_affine_btn_{image_key}"):
        try:
            st.session_state[image_key] = apply_affine(
                st.session_state.original_aug_image,
                shear=shear,
                scale=scale,
                translate_x=translate_x,
                translate_y=translate_y
            )
            # Success message removed
        except Exception as e:
            st.error(f"Error applying Affine Transformation: {e}")

def normalize_controls(image_key):
    """Controls for Normalize."""
    if st.button("Reapply Normalize", key=f"reapply_normalize_btn_{image_key}"):
        try:
            st.session_state[image_key] = apply_normalize(st.session_state.original_aug_image)
            # Success message removed
        except Exception as e:
            st.error(f"Error applying Normalize: {e}")
