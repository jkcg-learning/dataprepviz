import streamlit as st
import numpy as np
from PIL import Image
import cv2
from utils.helpers import apply_noise
import random

def image_augmentation_tab():
    st.header("🔄 Image Augmentation - Classical Methods")
    st.markdown("""
    Apply various classical augmentation techniques to diversify your dataset and improve model robustness.
    """)

    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_image = st.file_uploader("📁 Upload Image", type=['png', 'jpg', 'jpeg'])

        if uploaded_image is not None:
            image = load_image_cv2(uploaded_image)
            st.image(image, caption="📷 Original Image", use_column_width=True)

            st.subheader("⚙️ Augmentation Options")

            # Augmentation options
            rotation_angle = st.slider("🔄 Rotation Angle (degrees)", -180, 180, 0)
            scale_factor = st.slider("📏 Scale Factor", 0.5, 2.0, 1.0)
            add_noise = st.checkbox("🧂 Add Noise")
            noise_type = 'Gaussian' if add_noise else 'None'

            flip_horizontal = st.checkbox("↔️ Horizontal Flip")
            flip_vertical = st.checkbox("↕️ Vertical Flip")
            perspective_transform = st.checkbox("🔍 Perspective Transform")

            random_crop = st.checkbox("✂️ Random Crop")
            crop_size = st.slider("📐 Crop Size (%)", 10, 90, 80)

            if st.button("✅ Apply Augmentations"):
                augmented_images = []
                captions = []

                # Horizontal Flip
                if flip_horizontal:
                    flipped_h = cv2.flip(image, 1)
                    augmented_images.append(flipped_h)
                    captions.append("Horizontal Flip ↔️")

                # Vertical Flip
                if flip_vertical:
                    flipped_v = cv2.flip(image, 0)
                    augmented_images.append(flipped_v)
                    captions.append("Vertical Flip ↕️")

                # Rotation and Scaling
                if rotation_angle != 0 or scale_factor != 1.0:
                    height, width = image.shape[:2]
                    center = (width // 2, height // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, scale_factor)
                    rotated_scaled = cv2.warpAffine(image, rotation_matrix, (width, height))
                    augmented_images.append(rotated_scaled)
                    captions.append(f"Rotation {rotation_angle}° & Scale {scale_factor:.1f}")

                # Perspective Transform
                if perspective_transform:
                    height, width = image.shape[:2]
                    src_points = np.float32([
                        [0, 0],
                        [width - 1, 0],
                        [0, height - 1],
                        [width - 1, height - 1]
                    ])
                    offset = width * 0.1
                    dst_points = np.float32([
                        [offset, offset],
                        [width - offset, offset],
                        [offset, height - offset],
                        [width - offset, height - offset]
                    ])
                    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                    perspective = cv2.warpPerspective(image, matrix, (width, height))
                    augmented_images.append(perspective)
                    captions.append("Perspective Transform 🔍")

                # Random Crop
                if random_crop:
                    height, width = image.shape[:2]
                    crop_width = int(width * crop_size / 100)
                    crop_height = int(height * crop_size / 100)
                    start_x = random.randint(0, width - crop_width)
                    start_y = random.randint(0, height - crop_height)
                    cropped = image[start_y:start_y + crop_height, start_x:start_x + crop_width]
                    resized_cropped = cv2.resize(cropped, (width, height))
                    augmented_images.append(resized_cropped)
                    captions.append(f"Random Crop {crop_size}% ✂️")

                # Add Noise
                if add_noise:
                    noisy_img = apply_noise(image, noise_type='gaussian', noise_level=0.05)
                    augmented_images.append(noisy_img)
                    captions.append("Gaussian Noise 🧂")

                # Display Augmented Images
                with col2:
                    st.subheader("📈 Augmentation Results")
                    for img, caption in zip(augmented_images, captions):
                        st.image(img, caption=caption, use_column_width=True)

def load_image_cv2(uploaded_file):
    """Load an uploaded image file using OpenCV."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
