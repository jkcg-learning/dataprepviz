import streamlit as st
from modules.image.preprocessing import image_preprocessing_tab
from modules.image.augmentation import image_augmentation_tab  # Uncomment if available

def main():
    st.set_page_config(page_title="DataPrepViz", layout="wide")
    st.title("💾 Data Preprocessing & Augmentation :  Visualization Tool")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    data_type = st.sidebar.selectbox("Select Data Type", ["Image", "Text", "Audio", "3D"])
    operation = st.sidebar.selectbox("Select Operation", ["Preprocessing", "Augmentation"])

    if data_type == "Image":
        if operation == "Preprocessing":
            image_preprocessing_tab()
        elif operation == "Augmentation":
            image_augmentation_tab()  
            st.warning("🚧 Image Augmentation feature is under construction. Stay tuned!")
    else:
        st.warning("🚧 This feature is under construction. Stay tuned for updates!")

if __name__ == "__main__":
    main()




