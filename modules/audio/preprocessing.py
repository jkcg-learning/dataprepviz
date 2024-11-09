
import streamlit as st
import torchaudio
from utils.helpers import resample_audio, extract_mfcc, count_audio_tokens, encode_audio_to_wav, plot_waveform, plot_mfcc_heatmap
from io import BytesIO

def audio_preprocessing_tab():
    st.header("🎵 Audio Preprocessing")
    st.markdown("""
    Prepare your audio data using various preprocessing techniques to enhance quality and consistency.
    """)

    # Initialize session state for audio preprocessing
    if 'original_audio' not in st.session_state:
        st.session_state.original_audio = None
    if 'processed_audio' not in st.session_state:
        st.session_state.processed_audio = None
    if 'mfcc_features' not in st.session_state:
        st.session_state.mfcc_features = None
    if 'audio_token_count' not in st.session_state:
        st.session_state.audio_token_count = None

    # Audio Input
    uploaded_audio = st.file_uploader("📁 Upload an Audio File", type=['wav', 'mp3', 'flac'])

    if uploaded_audio is not None:
        # Load audio
        waveform, sample_rate = torchaudio.load(uploaded_audio)
        st.session_state.original_audio = waveform
        st.session_state.sample_rate = sample_rate

    if st.session_state.original_audio is not None:
        # Display Original Audio
        st.subheader("📄 Original Audio")
        buffer = encode_audio_to_wav(st.session_state.original_audio, st.session_state.sample_rate)
        st.audio(buffer, format='audio/wav')

        # Plot Original Waveform
        st.subheader("📈 Original Waveform")
        waveform_plot = plot_waveform(st.session_state.original_audio, st.session_state.sample_rate, title="Original Waveform")
        st.image(waveform_plot, use_column_width=True)

        st.markdown("---")

        # Resampling
        st.subheader("🔄 Resampling")
        target_sr = st.slider("Target Sampling Rate (Hz)", min_value=8000, max_value=48000, value=16000, step=1000)
        if st.button("Apply Resampling"):
            try:
                resampled_audio = resample_audio(st.session_state.original_audio, st.session_state.sample_rate, target_sr)
                st.session_state.processed_audio = resampled_audio
                st.session_state.sample_rate = target_sr
                st.success("✅ Resampling applied.")
            except Exception as e:
                st.error(f"Error applying resampling: {e}")

        # Display Processed Audio and Waveform
        if st.session_state.processed_audio is not None:
            st.subheader("🔄 Processed Audio")
            processed_buffer = encode_audio_to_wav(st.session_state.processed_audio, st.session_state.sample_rate)
            st.audio(processed_buffer, format='audio/wav')

            # Plot Processed Waveform
            st.subheader("📈 Processed Waveform")
            processed_waveform_plot = plot_waveform(st.session_state.processed_audio, st.session_state.sample_rate, title="Processed Waveform")
            st.image(processed_waveform_plot, use_column_width=True)

            # Download Processed Audio
            download_audio_button(st.session_state.processed_audio, 'Processed Audio', 'processed_audio.wav')

            st.markdown("---")

        # Feature Extraction
        st.subheader("📊 Feature Extraction (MFCC)")
        n_mfcc = st.number_input("Number of MFCC Features", min_value=10, max_value=60, value=40, step=1)
        if st.button("Extract MFCC Features"):
            try:
                mfcc = extract_mfcc(st.session_state.processed_audio if st.session_state.processed_audio is not None else st.session_state.original_audio, st.session_state.sample_rate, n_mfcc)
                st.session_state.mfcc_features = mfcc
                token_count = count_audio_tokens(st.session_state.mfcc_features)
                st.session_state.audio_token_count = token_count
                st.success("✅ MFCC features extracted.")
            except Exception as e:
                st.error(f"Error extracting MFCC features: {e}")

        # Display MFCC Features
        if st.session_state.mfcc_features is not None:
            st.subheader("📈 MFCC Features")
            # Plot MFCC Features as Heatmap
            fig = plot_mfcc_heatmap(st.session_state.mfcc_features, title="MFCC Features")
            st.pyplot(fig)

            st.write(f"**Number of Tokens:** {st.session_state.audio_token_count}")
            # Download MFCC as CSV
            download_audio_feature_button(st.session_state.mfcc_features, 'MFCC Features', 'mfcc_features.csv')

        st.markdown("---")

def download_audio_feature_button(feature, feature_name, file_name):
    """Provide a download button for audio features."""
    import pandas as pd
    # Flatten the tensor appropriately
    if feature.ndim == 3:
        # [channels, n_mfcc, time] -> [channels, n_mfcc * time]
        channels, n_mfcc, time = feature.shape
        df = pd.DataFrame(feature.numpy().reshape(channels, n_mfcc * time))
    elif feature.ndim == 4:
        # Handle batch dimension if present
        batch, channels, n_mfcc, time = feature.shape
        df = pd.DataFrame(feature.numpy().reshape(batch, channels * n_mfcc * time))
    else:
        # Handle other dimensions if necessary
        df = pd.DataFrame(feature.numpy())
    csv = df.to_csv(index=False)
    buf = BytesIO()
    buf.write(csv.encode('utf-8'))
    byte_im = buf.getvalue()

    st.download_button(
        label=f"📥 Download {feature_name}",
        data=byte_im,
        file_name=file_name,
        mime='text/csv'
    )

def download_audio_button(audio, technique_name, file_name):
    """Provide a download button for audio."""
    from utils.helpers import encode_audio_to_wav
    buffer = encode_audio_to_wav(audio, st.session_state.sample_rate)  # Use dynamic sample_rate
    st.download_button(
        label=f"📥 Download {technique_name}",
        data=buffer,
        file_name=file_name,
        mime='audio/wav'
    )
