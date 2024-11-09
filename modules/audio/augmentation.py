import streamlit as st
import torchaudio
from utils.helpers import (
    time_stretch, 
    pitch_shift, 
    add_background_noise, 
    random_gain, 
    random_silence, 
    count_audio_tokens,
    encode_audio_to_wav,
    plot_waveform
)
from io import BytesIO

def audio_augmentation_tab():
    st.header("🎶 Audio Augmentation")
    st.markdown("""
    Increase the diversity of your audio data using various augmentation techniques.
    """)

    # Initialize session state for audio augmentation
    if 'original_audio_aug' not in st.session_state:
        st.session_state.original_audio_aug = None
    if 'sample_rate_aug' not in st.session_state:
        st.session_state.sample_rate_aug = None
    # Initialize augmented audios
    augmented_keys = [
        'time_stretched_audio',
        'pitch_shifted_audio',
        'background_noisy_audio',
        'gain_adjusted_audio',
        'silence_inserted_audio'
    ]
    for key in augmented_keys:
        if key not in st.session_state:
            st.session_state[key] = None

    # Audio Input
    uploaded_audio = st.file_uploader("📁 Upload an Audio File for Augmentation", type=['wav', 'mp3', 'flac'])

    # Detect if new audio is uploaded
    if uploaded_audio is not None:
        # Load audio
        waveform, sample_rate = torchaudio.load(uploaded_audio)
        st.session_state.original_audio_aug = waveform
        st.session_state.sample_rate_aug = sample_rate
        # Reset augmented audios
        for key in augmented_keys:
            st.session_state[key] = None

    if st.session_state.original_audio_aug is not None:
        # Display Original Audio
        st.subheader("📄 Original Audio")
        # Encode audio to WAV and pass to st.audio
        buffer = encode_audio_to_wav(st.session_state.original_audio_aug, st.session_state.sample_rate_aug)
        st.audio(buffer, format='audio/wav')

        # Plot Original Waveform
        st.subheader("📈 Original Waveform")
        waveform_plot = plot_waveform(st.session_state.original_audio_aug, st.session_state.sample_rate_aug, title="Original Waveform")
        st.image(waveform_plot, use_column_width=True)

        st.markdown("---")

        # Define Augmentation Techniques
        augmentations = [
            {
                'name': 'Time Stretching',
                'result_key': 'time_stretched_audio',
                'controls': time_stretch_controls,
                'caption': '⏱️ Time Stretched Audio'
            },
            {
                'name': 'Pitch Shifting',
                'result_key': 'pitch_shifted_audio',
                'controls': pitch_shift_controls,
                'caption': '🎵 Pitch Shifted Audio'
            },
            {
                'name': 'Add Background Noise',
                'result_key': 'background_noisy_audio',
                'controls': add_background_noise_controls,
                'caption': '🌫️ Background Noisy Audio'
            },
            {
                'name': 'Random Gain Adjustment',
                'result_key': 'gain_adjusted_audio',
                'controls': gain_adjustment_controls,
                'caption': '🔊 Gain Adjusted Audio'
            },
            {
                'name': 'Random Silence Insertion',
                'result_key': 'silence_inserted_audio',
                'controls': silence_insertion_controls,
                'caption': '🤫 Silence Inserted Audio'
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
                        augmentation['controls']()
                        if st.session_state[augmentation['result_key']] is not None:
                            # Encode augmented audio and play
                            augmented_buffer = encode_audio_to_wav(st.session_state[augmentation['result_key']], st.session_state.sample_rate_aug)
                            st.audio(augmented_buffer, format='audio/wav')
                            # Plot Augmented Waveform
                            st.subheader(f"📈 {augmentation['name']} Waveform")
                            augmented_waveform_plot = plot_waveform(st.session_state[augmentation['result_key']], st.session_state.sample_rate_aug, title=f"{augmentation['name']} Waveform")
                            st.image(augmented_waveform_plot, use_column_width=True)
                            # Token Count
                            token_count = count_audio_tokens(st.session_state[augmentation['result_key']])
                            st.write(f"**Number of Tokens:** {token_count}")
                            # Download Button
                            download_audio_button(
                                st.session_state[augmentation['result_key']], 
                                augmentation['name'], 
                                f"{augmentation['name'].lower().replace(' ', '_')}_audio.wav"
                            )

        st.markdown("---")

def time_stretch_controls():
    """Controls for Time Stretching."""
    rate = st.slider(
        "Stretch Rate",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Factor by which to stretch the audio. <1.0 slows down, >1.0 speeds up."
    )
    if st.button("Apply Time Stretching", key="apply_time_stretching"):
        try:
            augmented_audio = time_stretch(st.session_state.original_audio_aug, rate)
            st.session_state.time_stretched_audio = augmented_audio
            st.success("✅ Time Stretching applied.")
        except Exception as e:
            st.error(f"Error applying Time Stretching: {e}")

def pitch_shift_controls():
    """Controls for Pitch Shifting."""
    n_steps = st.number_input(
        "Pitch Shift (Semitones)",
        min_value=-12,
        max_value=12,
        value=0,
        step=1,
        help="Number of semitones to shift the pitch. Negative for lower pitch, positive for higher pitch."
    )
    if st.button("Apply Pitch Shifting", key="apply_pitch_shift"):
        try:
            augmented_audio = pitch_shift(st.session_state.original_audio_aug, st.session_state.sample_rate_aug, n_steps)
            st.session_state.pitch_shifted_audio = augmented_audio
            st.success("✅ Pitch Shifting applied.")
        except Exception as e:
            st.error(f"Error applying Pitch Shifting: {e}")

def add_background_noise_controls():
    """Controls for Adding Background Noise."""
    noise_factor = st.slider(
        "Noise Factor",
        min_value=0.0,
        max_value=0.1,
        value=0.005,
        step=0.001,
        help="Factor by which to add background noise."
    )
    if st.button("Apply Background Noise", key="apply_background_noise"):
        try:
            augmented_audio = add_background_noise(st.session_state.original_audio_aug, noise_factor)
            st.session_state.background_noisy_audio = augmented_audio
            st.success("✅ Background Noise added.")
        except Exception as e:
            st.error(f"Error adding Background Noise: {e}")

def gain_adjustment_controls():
    """Controls for Random Gain Adjustment."""
    gain_min = st.number_input(
        "Minimum Gain",
        min_value=0.5,
        max_value=1.0,
        value=0.8,
        step=0.1,
        help="Minimum gain factor."
    )
    gain_max = st.number_input(
        "Maximum Gain",
        min_value=1.0,
        max_value=2.0,
        value=1.2,
        step=0.1,
        help="Maximum gain factor."
    )
    if st.button("Apply Gain Adjustment", key="apply_gain_adjustment"):
        try:
            augmented_audio = random_gain(st.session_state.original_audio_aug, gain_min, gain_max)
            st.session_state.gain_adjusted_audio = augmented_audio
            st.success("✅ Gain Adjustment applied.")
        except Exception as e:
            st.error(f"Error applying Gain Adjustment: {e}")

def silence_insertion_controls():
    """Controls for Random Silence Insertion."""
    min_silence = st.slider(
        "Minimum Silence Duration (seconds)",
        min_value=0.1,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="Minimum duration of silence to insert."
    )
    max_silence = st.slider(
        "Maximum Silence Duration (seconds)",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
        help="Maximum duration of silence to insert."
    )
    if st.button("Apply Silence Insertion", key="apply_silence_insertion"):
        try:
            augmented_audio = random_silence(st.session_state.original_audio_aug, min_silence, max_silence)
            st.session_state.silence_inserted_audio = augmented_audio
            st.success("✅ Silence Insertion applied.")
        except Exception as e:
            st.error(f"Error applying Silence Insertion: {e}")

def download_audio_button(audio, technique_name, file_name):
    """Provide a download button for augmented audio."""
    buffer = encode_audio_to_wav(audio, st.session_state.sample_rate_aug)  # Use dynamic sample_rate
    st.download_button(
        label=f"📥 Download {technique_name}",
        data=buffer,
        file_name=file_name,
        mime='audio/wav'
    )
