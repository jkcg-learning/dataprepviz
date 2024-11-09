
import streamlit as st
from utils.helpers import load_mesh, normalize_mesh, center_mesh, plot_mesh

def threeD_preprocessing_tab():
    st.header("📐 3D Data Preprocessing")
    st.markdown("""
    Prepare your 3D data using various preprocessing techniques to enhance quality and consistency.
    """)

    # Initialize session state
    if 'original_mesh' not in st.session_state:
        st.session_state.original_mesh = None
    if 'processed_mesh' not in st.session_state:
        st.session_state.processed_mesh = None

    # File Uploader
    uploaded_file = st.file_uploader("📁 Upload a 3D Model File", type=['obj', 'stl', 'ply'])

    if uploaded_file is not None:
        # Extract the file extension
        file_extension = uploaded_file.name.split('.')[-1]
        try:
            # Load the mesh with the specified file type
            mesh = load_mesh(uploaded_file, str(file_extension))
            st.session_state.original_mesh = mesh
        except ValueError as e:
            st.error(f"Error loading 3D model: {e}")

    if st.session_state.original_mesh is not None:
        # Display Original Mesh
        st.subheader("📄 Original Mesh")
        fig = plot_mesh(st.session_state.original_mesh, title="Original Mesh")
        st.plotly_chart(fig)

        st.markdown("---")

        # Preprocessing Options
        st.subheader("⚙️ Preprocessing Techniques")
        if st.button("Normalize Mesh"):
            try:
                mesh = normalize_mesh(st.session_state.original_mesh.copy())
                st.session_state.processed_mesh = mesh
                st.success("✅ Mesh normalized.")
            except Exception as e:
                st.error(f"Error normalizing mesh: {e}")

        if st.button("Center Mesh"):
            try:
                mesh = center_mesh(st.session_state.original_mesh.copy())
                st.session_state.processed_mesh = mesh
                st.success("✅ Mesh centered.")
            except Exception as e:
                st.error(f"Error centering mesh: {e}")

        # Display Processed Mesh
        if st.session_state.processed_mesh is not None:
            st.subheader("🔄 Processed Mesh")
            fig = plot_mesh(st.session_state.processed_mesh, title="Processed Mesh")
            st.plotly_chart(fig)
            # Download Processed Mesh
            download_mesh_button(st.session_state.processed_mesh, "Processed Mesh", "processed_mesh.obj")

def download_mesh_button(mesh, technique_name, file_name):
    """Provide a download button for the mesh."""
    buffer = mesh.export(file_type='obj')
    st.download_button(
        label=f"📥 Download {technique_name}",
        data=buffer,
        file_name=file_name,
        mime='text/plain'
    )
