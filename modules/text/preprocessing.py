import streamlit as st
from utils.helpers import apply_text_preprocessing, count_tokens, tokenize_text, pad_truncate_text, embed_text
from io import BytesIO

def text_preprocessing_tab():
    st.header("📄 Text Preprocessing")
    st.markdown("""
    Enhance and prepare your text data using various preprocessing techniques.
    """)
    
    # Initialize session state for text preprocessing
    if 'original_text_pre' not in st.session_state:
        st.session_state.original_text_pre = None
    # Initialize preprocessed texts
    preprocessed_keys = [
        'tokenized_text',
        'embedded_text',
        'token_count'
    ]
    for key in preprocessed_keys:
        if key not in st.session_state:
            st.session_state[key] = None
    
    # Text Input
    text_input = st.text_area("📝 Enter Text for Preprocessing", height=200)
    
    # Alternatively, File Uploader
    uploaded_text = st.file_uploader("📁 Or Upload a Text File", type=['txt'])
    
    if uploaded_text is not None:
        text_input = uploaded_text.read().decode('utf-8')
    
    if text_input:
        # Store original text in session state
        if st.session_state.original_text_pre is None:
            st.session_state.original_text_pre = text_input
        
        # Display Original Text
        st.subheader("📄 Original Text")
        st.write(st.session_state.original_text_pre)
        
        st.markdown("---")
        
        # Apply Preprocessing with default parameters
        if st.session_state.tokenized_text is None:
            preprocessed = apply_text_preprocessing(st.session_state.original_text_pre, max_length=50)
            st.session_state.tokenized_text = preprocessed['tokens']
            st.session_state.embedded_text = preprocessed['embedded']
            st.session_state.token_count = preprocessed['token_count']
        
        # Display Preprocessed Texts
        preprocessings = [
            {
                'name': 'Tokenization',
                'result_key': 'tokenized_text',
                'controls': tokenization_controls,
                'caption': '🔠 Tokenized Text'
            },
            {
                'name': 'Padding/Truncating',
                'result_key': 'tokenized_text',
                'controls': padding_truncating_controls,
                'caption': '📏 Padded/Truncated Text'
            },
            {
                'name': 'Embedding',
                'result_key': 'embedded_text',
                'controls': embedding_controls,
                'caption': '🔗 Embedded Text'
            },
            {
                'name': 'Token Counting',
                'result_key': 'token_count',
                'controls': token_count_display,
                'caption': '🔢 Token Count'
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
                        if st.session_state[preprocessing['result_key']] is not None:
                            if preprocessing['name'] == 'Embedding':
                                st.write(st.session_state[preprocessing['result_key']])
                                # Download Button
                                download_text_button(preprocessing['result_key'], preprocessing['name'], preprocessing['caption'])
                            elif preprocessing['name'] == 'Token Counting':
                                st.write(f"**Number of Tokens:** {st.session_state[preprocessing['result_key']]}")
                            else:
                                st.write(st.session_state[preprocessing['result_key']])
                                # Download Button
                                download_text_button(preprocessing['result_key'], preprocessing['name'], preprocessing['caption'])
        
        st.markdown("---")

def tokenization_controls():
    """Controls for Tokenization."""
    if st.button("Reapply Tokenization", key="reapply_tokenization_btn"):
        try:
            tokens = tokenize_text(st.session_state.original_text_pre)
            st.session_state.tokenized_text = tokens
            # Update token count
            st.session_state.token_count = count_tokens(st.session_state.original_text_pre)
            # st.success("✅ Tokenization applied.")
        except Exception as e:
            st.error(f"Error applying Tokenization: {e}")

def padding_truncating_controls():
    """Controls for Padding/Truncating."""
    max_length = st.number_input(
        "Max Sequence Length",
        min_value=10,
        max_value=500,
        value=50,
        step=1,
        key="max_length_input",
        help="Define the maximum sequence length for padding/truncating."
    )
    if st.button("Reapply Padding/Truncating", key="reapply_padding_truncating_btn"):
        try:
            tokens = pad_truncate_text(st.session_state.tokenized_text, max_length=int(max_length))
            st.session_state.tokenized_text = tokens
            # Update token count
            st.session_state.token_count = count_tokens(' '.join(tokens))
            # st.success("✅ Padding/Truncating applied.")
        except Exception as e:
            st.error(f"Error applying Padding/Truncating: {e}")

def embedding_controls():
    """Controls for Embedding."""
    if st.button("Reapply Embedding", key="reapply_embedding_btn"):
        try:
            embedded = embed_text(st.session_state.tokenized_text)
            st.session_state.embedded_text = embedded
            # st.success("✅ Embedding applied.")
        except Exception as e:
            st.error(f"Error applying Embedding: {e}")

def token_count_display():
    """Display Token Count."""
    token_count = st.session_state.token_count
    st.write(f"**Number of Tokens:** {token_count}")

def download_text_button(state_key, technique_name, caption):
    """Provide a download button for the processed text."""
    if state_key == 'embedded_text':
        text = ', '.join(map(str, st.session_state[state_key]))
        file_ext = 'csv'
        mime_type = 'text/csv'
        file_name = f"{technique_name.lower().replace(' ', '_')}_text.csv"
    else:
        text = ' '.join(st.session_state[state_key])
        file_ext = 'txt'
        mime_type = 'text/plain'
        file_name = f"{technique_name.lower().replace(' ', '_')}_text.{file_ext}"
    
    buf = BytesIO()
    buf.write(text.encode('utf-8'))
    byte_im = buf.getvalue()
    
    st.download_button(
        label=f"📥 Download {technique_name} Text",
        data=byte_im,
        file_name=file_name,
        mime=mime_type
    )
