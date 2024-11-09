# modules/data_processing/text_augmentation.py

import streamlit as st
from utils.helpers import apply_text_augmentation, count_tokens
from io import BytesIO

def text_augmentation_tab():
    st.header("🔄 Text Augmentation")
    st.markdown("""
    Increase the diversity of your text data using various augmentation techniques.
    """)
    
    # Initialize session state for text augmentation
    if 'original_text_aug' not in st.session_state:
        st.session_state.original_text_aug = None
    # Initialize augmented texts
    augmented_keys = [
        'synonym_replaced_text',
        'randomly_inserted_text',
        'randomly_deleted_text',
        # 'back_translated_text'
    ]
    for key in augmented_keys:
        if key not in st.session_state:
            st.session_state[key] = None
    
    # Text Input
    text_input = st.text_area("📝 Enter Text for Augmentation", height=200)
    
    # Alternatively, File Uploader
    uploaded_text = st.file_uploader("📁 Or Upload a Text File", type=['txt'])
    
    # Detect if new text is uploaded or entered
    if uploaded_text is not None:
        new_text = uploaded_text.read().decode('utf-8')
        if st.session_state.original_text_aug != new_text:
            # New text detected, reset augmented texts
            st.session_state.original_text_aug = new_text
            for key in augmented_keys:
                st.session_state[key] = None
            text_input = new_text
    elif text_input != st.session_state.original_text_aug:
        # Text area content changed, reset augmented texts
        st.session_state.original_text_aug = text_input
        for key in augmented_keys:
            st.session_state[key] = None
    
    if st.session_state.original_text_aug:
        # Display Original Text
        st.subheader("📄 Original Text")
        st.write(st.session_state.original_text_aug)
        
        st.markdown("---")
        
        # Apply Augmentation with default parameters if not already applied
        if st.session_state.synonym_replaced_text is None:
            st.session_state.synonym_replaced_text = apply_text_augmentation(
                st.session_state.original_text_aug, 
                augmentation_type='Synonym Replacement', 
                n=1
            )
        if st.session_state.randomly_inserted_text is None:
            st.session_state.randomly_inserted_text = apply_text_augmentation(
                st.session_state.original_text_aug, 
                augmentation_type='Random Insertion', 
                n=1
            )
        if st.session_state.randomly_deleted_text is None:
            st.session_state.randomly_deleted_text = apply_text_augmentation(
                st.session_state.original_text_aug, 
                augmentation_type='Random Deletion', 
                p=0.1
            )
        # if st.session_state.back_translated_text is None:
        #    st.session_state.back_translated_text = apply_text_augmentation(
        #        st.session_state.original_text_aug, 
        #        augmentation_type='Back Translation'
        #    )
        
        # Display Augmented Texts
        augmentations = [
            {
                'name': 'Synonym Replacement',
                'result_key': 'synonym_replaced_text',
                'controls': synonym_replacement_controls,
                'caption': '🔄 Synonym Replaced Text'
            },
            {
                'name': 'Random Insertion',
                'result_key': 'randomly_inserted_text',
                'controls': random_insertion_controls,
                'caption': '➕ Randomly Inserted Text'
            },
            {
                'name': 'Random Deletion',
                'result_key': 'randomly_deleted_text',
                'controls': random_deletion_controls,
                'caption': '➖ Randomly Deleted Text'
            },
            # {
            #     'name': 'Back Translation',
            #     'result_key': 'back_translated_text',
            #     'controls': back_translation_controls,
            #     'caption': '🔁 Back Translated Text'
            # }
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
                            st.write(st.session_state[augmentation['result_key']])
                            # Token Count
                            token_count = count_tokens(st.session_state[augmentation['result_key']])
                            st.write(f"**Number of Tokens:** {token_count}")
                            # Download Button
                            download_text_button(
                                augmentation['result_key'], 
                                augmentation['name'], 
                                augmentation['caption']
                            )
        
        st.markdown("---")

def synonym_replacement_controls():
    """Controls for Synonym Replacement."""
    n = st.number_input(
        "Number of Synonyms to Replace",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        key="synonym_replacement_n",
        help="Specify how many words to replace with synonyms."
    )
    if st.button("Reapply Synonym Replacement", key="reapply_synonym_replacement_btn"):
        try:
            augmented_text = apply_text_augmentation(
                st.session_state.original_text_aug, 
                augmentation_type='Synonym Replacement', 
                n=int(n)
            )
            st.session_state.synonym_replaced_text = augmented_text
            st.success("✅ Synonym Replacement applied.")
        except Exception as e:
            st.error(f"Error applying Synonym Replacement: {e}")

def random_insertion_controls():
    """Controls for Random Insertion."""
    n = st.number_input(
        "Number of Synonyms to Insert",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        key="random_insertion_n",
        help="Specify how many synonyms to insert into the text."
    )
    if st.button("Reapply Random Insertion", key="reapply_random_insertion_btn"):
        try:
            augmented_text = apply_text_augmentation(
                st.session_state.original_text_aug, 
                augmentation_type='Random Insertion', 
                n=int(n)
            )
            st.session_state.randomly_inserted_text = augmented_text
            st.success("✅ Random Insertion applied.")
        except Exception as e:
            st.error(f"Error applying Random Insertion: {e}")

def random_deletion_controls():
    """Controls for Random Deletion."""
    p = st.slider(
        "Deletion Probability (p)",
        0.0,
        1.0,
        0.1,
        0.05,
        key="random_deletion_p",
        help="Probability of deleting each word in the text."
    )
    if st.button("Reapply Random Deletion", key="reapply_random_deletion_btn"):
        try:
            augmented_text = apply_text_augmentation(
                st.session_state.original_text_aug, 
                augmentation_type='Random Deletion', 
                p=p
            )
            st.session_state.randomly_deleted_text = augmented_text
            st.success("✅ Random Deletion applied.")
        except Exception as e:
            st.error(f"Error applying Random Deletion: {e}")

def back_translation_controls():
    """Controls for Back Translation."""
    if st.button("Reapply Back Translation", key="reapply_back_translation_btn"):
        try:
            augmented_text = apply_text_augmentation(
                st.session_state.original_text_aug, 
                augmentation_type='Back Translation'
            )
            st.session_state.back_translated_text = augmented_text
            st.success("✅ Back Translation applied.")
        except Exception as e:
            st.error(f"Error applying Back Translation: {e}")

def download_text_button(state_key, technique_name, caption):
    """Provide a download button for the augmented text."""
    text = st.session_state[state_key]
    if technique_name == 'Embedding':
        # For embeddings, download as CSV
        text = ', '.join(map(str, st.session_state[state_key]))
        file_ext = 'csv'
        mime_type = 'text/csv'
    else:
        # For text augmentations, download as TXT
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
