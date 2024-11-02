import streamlit as st
from modules.image.preprocessing import image_preprocessing_tab
from modules.image.augmentation import image_augmentation_tab
# from modules.text.preprocessing import text_preprocessing_tab
# from modules.text.augmentation import text_augmentation_tab
# from modules.audio.preprocessing import audio_preprocessing_tab
# from modules.audio.augmentation import audio_augmentation_tab

def main():
    st.set_page_config(layout="wide")
    st.title("Data Preprocessing and Augmentation Visualization")

    data_type = st.sidebar.selectbox("Select Data Type", ["Image", "Text", "Audio", "3D"])
    operation = st.sidebar.selectbox("Select Operation", ["Preprocessing", "Augmentation"])
    method = st.sidebar.selectbox("Select Method", ["Classical", "Deep Learning"])

    if data_type == "Image":
        if operation == "Preprocessing":
            image_preprocessing_tab(method)
        else:
            image_augmentation_tab(method)

if __name__ == "__main__":
    main()
