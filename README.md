# DataPrepViz

## 📊 **Overview**

**DataPrepViz** is an interactive web application designed to streamline the data preprocessing and augmentation workflow for various data types. Built with Streamlit, it offers a user-friendly interface to apply a wide range of transformations to your data, enhancing its quality and suitability for machine learning models. Currently focused on **Image** and **Text** data, DataPrepViz is architected to seamlessly extend support to other data types such as **Audio** and **3D** in the future.

## 🚀 **Features**

### 🖼️ **Image Preprocessing**

Enhance and prepare your images using a suite of preprocessing techniques:

- **Resize**
  - Adjust the dimensions of your image to the desired width and height.

- **Contrast & Brightness Adjustment**
  - Modify the contrast (α) and brightness (β) levels to improve image visibility.

- **Blurring**
  - **Gaussian Blur**: Smooth images using a Gaussian kernel.
  - **Median Blur**: Reduce noise with a median filter.
  - **Bilateral Blur**: Preserve edges while blurring.

- **Noise Addition**
  - **Gaussian Noise**: Add Gaussian-distributed noise.
  - **Salt & Pepper Noise**: Introduce salt and pepper noise for robustness testing.

- **Edge Detection**
  - **Canny Edge Detection**: Detect edges using the Canny algorithm.
  - **Sobel Edge Detection**: Highlight edges with the Sobel operator.
  - **Laplacian Edge Detection**: Capture edge details using the Laplacian method.

- **Color Space Conversion**
  - Convert images between RGB, HSV, LAB, and YCrCb color spaces.

- **Grayscale Conversion**
  - Transform color images into grayscale for simplified analysis.

- **Normalization**
  - Normalize pixel values to a standard range [0, 255] for consistency.

### 🔄 **Image Augmentation**

Enhance your dataset's diversity with powerful augmentation techniques:

- **Horizontal Flip**
  - Flip images horizontally to simulate mirror views.

- **Vertical Flip**
  - Flip images vertically to account for upside-down scenarios.

- **Rotate**
  - Rotate images by specified angles to introduce rotational invariance.

- **Random Resized Crop**
  - Crop and resize images randomly to focus on different regions.

- **Color Jitter**
  - Randomly alter brightness, contrast, saturation, and hue to mimic varying lighting conditions.

- **Affine Transformation**
  - Apply shear, scaling, and translation to distort images realistically.

- **Normalize**
  - Standardize images using ImageNet statistics for better model compatibility.

### 📄 **Text Preprocessing**

Enhance and prepare your textual data using various preprocessing techniques:

- **Tokenization**
  - Split text into individual tokens (words).

- **Padding/Truncating**
  - Ensure uniform text length by padding shorter texts with `<PAD>` tokens or truncating longer texts to a specified maximum length.

- **Embedding**
  - Convert tokens into numerical vectors using pre-trained BERT embeddings.

- **Token Counting**
  - Count the number of tokens using `tiktoken` for efficient processing.

### ✍️ **Text Augmentation**

Increase the diversity of your text data using various augmentation techniques:

- **Synonym Replacement**
  - Replace a specified number of words with their synonyms to enhance text diversity.

- **Random Insertion**
  - Insert synonyms of random words into the text.

- **Random Deletion**
  - Remove words from the text based on a specified probability.

## 📥 **Download Transformed Data**

- **Images:**
  - Easily download any preprocessed or augmented image in your preferred format (**PNG** or **JPEG**) with just a click.

- **Text:**
  - Download processed and augmented text data as `.txt` files.
  - Download embedded text data as `.csv` files for compatibility with machine learning models.

## ⚡ **Performance Optimizations**

- **Caching Mechanisms:**
  - Leveraging Streamlit's caching (`@st.cache_data`) to ensure efficient processing and rapid response times, especially with large datasets.
