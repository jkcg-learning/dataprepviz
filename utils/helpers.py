
import numpy as np
import cv2
import albumentations as A
import streamlit as st
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import tiktoken
from transformers import BertTokenizer
import torchaudio
import torch
from io import BytesIO
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure NLTK data is downloaded
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')


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

# ======================
# Text Preprocessing Functions
# ======================

@st.cache_data
def tokenize_text(text):
    """Tokenize the input text into tokens."""
    tokens = word_tokenize(text)
    return tokens

@st.cache_data
def pad_truncate_text(tokens, max_length=50):
    """Pad or truncate the token list to a fixed length."""
    if len(tokens) < max_length:
        tokens += ['<PAD>'] * (max_length - len(tokens))
    else:
        tokens = tokens[:max_length]
    return tokens

@st.cache_data
def embed_text(tokens):
    """Embed tokens using pre-trained BERT embeddings."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    embedded = tokenizer.convert_tokens_to_ids(tokens)
    return embedded

@st.cache_data
def count_tokens(text):
    """Count the number of tokens in the text using tiktoken."""
    encoding = tiktoken.get_encoding("gpt2")
    tokens = encoding.encode(text)
    return len(tokens)

# ======================
# Text Augmentation Functions
# ======================

def get_synonyms(word):
    """Retrieve synonyms for a given word using WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ').lower()
            if synonym != word.lower():
                synonyms.add(synonym)
    return list(synonyms)

def synonym_replacement(text, n=1):
    """Replace n words in the text with their synonyms."""
    tokens = word_tokenize(text)
    new_tokens = tokens.copy()
    words = [word for word in tokens if wordnet.synsets(word)]
    if len(words) == 0:
        return text
    n = min(n, len(words))
    words_to_replace = np.random.choice(words, n, replace=False)
    for word in words_to_replace:
        synonyms = get_synonyms(word)
        if synonyms:
            synonym = np.random.choice(synonyms)
            new_tokens = [synonym if w == word else w for w in new_tokens]
    return ' '.join(new_tokens)

def random_insertion(text, n=1):
    """Insert n random synonyms into the text."""
    tokens = word_tokenize(text)
    new_tokens = tokens.copy()
    for _ in range(n):
        add_synonym(new_tokens)
    return ' '.join(new_tokens)

def add_synonym(tokens):
    """Add a synonym of a random word in the tokens."""
    words = [word for word in tokens if wordnet.synsets(word)]
    if not words:
        return tokens
    word = np.random.choice(words)
    synonyms = get_synonyms(word)
    if synonyms:
        synonym = np.random.choice(synonyms)
        insert_idx = np.random.randint(0, len(tokens) + 1)
        tokens.insert(insert_idx, synonym)
    return tokens

def random_deletion(text, p=0.1):
    """Randomly delete words from the text with probability p."""
    tokens = word_tokenize(text)
    if len(tokens) == 0:
        return text
    new_tokens = [word for word in tokens if np.random.rand() > p]
    if len(new_tokens) == 0:
        return np.random.choice(tokens)
    return ' '.join(new_tokens)

def back_translation(text, src_lang='en', tgt_lang='fr'):
    """
    Perform back translation by translating the text to another language and back.
    Requires external translation API or library.
    Placeholder function as implementation depends on the chosen library/API.
    """
    # Implement using a translation library like googletrans
    # from googletrans import Translator
    # translator = Translator()
    # translated = translator.translate(text, src=src_lang, dest=tgt_lang).text
    # back_translated = translator.translate(translated, src=tgt_lang, dest=src_lang).text
    #return back_translated
    return text

@st.cache_data
def apply_text_preprocessing(text, max_length=50):
    """Apply all preprocessing steps to the text."""
    tokens = tokenize_text(text)
    tokens = pad_truncate_text(tokens, max_length=max_length)
    embedded = embed_text(tokens)
    token_count = count_tokens(text)
    return {
        'tokens': tokens,
        'embedded': embedded,
        'token_count': token_count
    }

@st.cache_data
def apply_text_augmentation(text, augmentation_type='Synonym Replacement', n=1, p=0.1):
    """Apply specified augmentation to the text."""
    if augmentation_type == 'Synonym Replacement':
        return synonym_replacement(text, n)
    elif augmentation_type == 'Random Insertion':
        return random_insertion(text, n)
    elif augmentation_type == 'Random Deletion':
        return random_deletion(text, p)
    # elif augmentation_type == 'Back Translation':
    #    return back_translation(text)
    else:
        return text
    
# ======================
# Audio Preprocessing Functions
# ======================

@st.cache_data
def resample_audio(_audio, orig_sr, target_sr=16000):
    """Resample audio to a target sampling rate."""
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
    resampled_audio = resampler(_audio)
    return resampled_audio

@st.cache_data
def extract_mfcc(_audio, sample_rate=16000, n_mfcc=40):
    """Extract MFCC features from audio."""
    n_mels = 23  # Ensure n_mfcc <= n_mels
    if n_mfcc > n_mels:
        raise ValueError(f"Number of MFCC coefficients ({n_mfcc}) cannot exceed number of mel bins ({n_mels}).")
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            'n_fft': 400,
            'hop_length': 160,
            'n_mels': n_mels,
            'center': False
        }
    )
    mfcc = mfcc_transform(_audio)
    return mfcc

@st.cache_data
def count_audio_tokens(_audio):
    """Count the number of audio tokens using tiktoken."""
    encoding = tiktoken.get_encoding("gpt2")
    tokens = encoding.encode(str(_audio))
    return len(tokens)

# ======================
# Audio Augmentation Functions
# ======================

def time_stretch(audio, rate=1.0):
    """Stretch the audio in time by a given rate using torchaudio's sox_effects."""
    effects = [['tempo', str(rate)]]
    augmented_audio, sample_rate = torchaudio.sox_effects.apply_effects_tensor(audio, 16000, effects)
    return augmented_audio

def pitch_shift(audio, sample_rate=16000, n_steps=0):
    """Shift the pitch of the audio by n_steps semitones using torchaudio's sox_effects."""
    # SoX 'pitch' effect uses cents; 1 semitone = 100 cents
    shift_cents = n_steps * 100
    effects = [['pitch', str(shift_cents)], ['rate', str(sample_rate)]]
    augmented_audio, sample_rate = torchaudio.sox_effects.apply_effects_tensor(audio, sample_rate, effects)
    return augmented_audio

def add_background_noise(signal, noise_factor=0.005):
    """Add background noise to the audio signal."""
    noise = torch.randn_like(signal)
    augmented_signal = signal + noise_factor * noise
    return augmented_signal

def random_gain(audio, gain_min=0.8, gain_max=1.2):
    """Randomly adjust the gain of the audio signal."""
    gain = np.random.uniform(gain_min, gain_max)
    return audio * gain

def random_silence(audio, min_silence=0.1, max_silence=0.5):
    """Insert random silence into the audio signal."""
    silence_duration = np.random.uniform(min_silence, max_silence)
    num_channels = audio.shape[0]  # Assuming shape is [channels, samples]
    silence = torch.zeros((num_channels, int(silence_duration * 16000)))
    insert_position = np.random.randint(0, audio.shape[1])
    augmented_audio = torch.cat((audio[:, :insert_position], silence, audio[:, insert_position:]), dim=1)
    return augmented_audio

# ======================
# Audio Encoding Function
# ======================

def encode_audio_to_wav(audio_tensor, sample_rate):
    """
    Encode a Torch audio tensor to WAV format and return as BytesIO object.

    Parameters:
    - audio_tensor (torch.Tensor): Audio data tensor with shape [channels, samples].
    - sample_rate (int): Sampling rate of the audio.

    Returns:
    - BytesIO: Byte stream of the encoded WAV audio.
    """
    buffer = BytesIO()
    # Transpose the tensor to [samples, channels] if necessary
    if audio_tensor.ndim == 2:
        audio = audio_tensor.numpy().T
    else:
        audio = audio_tensor.numpy()
    sf.write(buffer, audio, sample_rate, format='WAV')
    buffer.seek(0)
    return buffer

# ======================
# Waveform Plotting Function
# ======================

def plot_waveform(audio_tensor, sample_rate, title="Waveform"):
    """
    Plot the waveform of an audio tensor.

    Parameters:
    - audio_tensor (torch.Tensor): Audio data tensor with shape [channels, samples].
    - sample_rate (int): Sampling rate of the audio.
    - title (str): Title of the plot.

    Returns:
    - BytesIO: Byte stream of the plotted waveform image.
    """
    plt.figure(figsize=(10, 4))
    for channel in audio_tensor:
        plt.plot(channel.numpy(), alpha=0.5)
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

# ======================
# MFCC Feature Plotting Function
# ======================

def plot_mfcc_heatmap(mfcc_tensor, title="MFCC Features"):
    """
    Plot the MFCC features as a heatmap.

    Parameters:
    - mfcc_tensor (torch.Tensor): MFCC features tensor with shape [channels, n_mfcc, time].
    - title (str): Title of the plot.

    Returns:
    - matplotlib.figure.Figure: The matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    mfcc_np = mfcc_tensor.numpy()
    for channel in range(mfcc_np.shape[0]):
        sns.heatmap(mfcc_np[channel], ax=ax, cmap='viridis', cbar=True)
    ax.set_title(title)
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("MFCC Coefficients")
    plt.tight_layout()
    return fig