import streamlit as st
from utils.helpers import load_mesh, random_rotation, scale_mesh, add_noise, plot_mesh

def threeD_augmentation_tab():
    st.header("🎲 3D Data Augmentation")
    st.markdown("""
    Increase the diversity of your 3D data using various augmentation techniques.
    """)

    # Initialize session state
    if 'original_mesh_aug' not in st.session_state:
        st.session_state.original_mesh_aug = None
    # Initialize augmented meshes
    augmented_keys = [
        'rotated_mesh',
        'scaled_mesh',
        'noisy_mesh'
    ]
    for key in augmented_keys:
        if key not in st.session_state:
            st.session_state[key] = None

    # File Uploader
    uploaded_file = st.file_uploader("📁 Upload a 3D Model File", type=['obj', 'stl', 'ply'])

    if uploaded_file is not None:
        # Extract the file extension
        file_extension = uploaded_file.name.split('.')[-1]
        try:
            # Load the mesh with the specified file type
            mesh = load_mesh(uploaded_file, str(file_extension))
            st.session_state.original_mesh_aug = mesh
        except ValueError as e:
            st.error(f"Error loading 3D model: {e}")

    if st.session_state.original_mesh_aug is not None:
        # Display Original Mesh
        st.subheader("📄 Original Mesh")
        fig = plot_mesh(st.session_state.original_mesh_aug, title="Original Mesh")
        st.plotly_chart(fig)

        st.markdown("---")

        # Augmentation Techniques
        st.subheader("🔧 Augmentation Techniques")

        # Random Rotation
        st.markdown("### 🔄 Random Rotation")
        if st.button("Apply Random Rotation"):
            try:
                mesh = random_rotation(st.session_state.original_mesh_aug.copy())
                st.session_state.rotated_mesh = mesh
                st.success("✅ Random rotation applied.")
            except Exception as e:
                st.error(f"Error applying random rotation: {e}")
        if st.session_state.rotated_mesh is not None:
            fig = plot_mesh(st.session_state.rotated_mesh, title="Rotated Mesh")
            st.plotly_chart(fig)
            download_mesh_button(st.session_state.rotated_mesh, "Rotated Mesh", "rotated_mesh.obj")

        # Scaling
        st.markdown("### 📏 Scaling")
        scale_factor = st.slider("Scale Factor", min_value=0.5, max_value=1.5, value=1.0, step=0.1)
        if st.button("Apply Scaling"):
            try:
                mesh = scale_mesh(st.session_state.original_mesh_aug.copy(), scale_factor)
                st.session_state.scaled_mesh = mesh
                st.success("✅ Scaling applied.")
            except Exception as e:
                st.error(f"Error applying scaling: {e}")
        if st.session_state.scaled_mesh is not None:
            fig = plot_mesh(st.session_state.scaled_mesh, title="Scaled Mesh")
            st.plotly_chart(fig)
            download_mesh_button(st.session_state.scaled_mesh, "Scaled Mesh", "scaled_mesh.obj")

        # Adding Noise
        st.markdown("### 🌪️ Adding Noise")
        noise_level = st.slider("Noise Level", min_value=0.0, max_value=0.1, value=0.01, step=0.005)
        if st.button("Apply Noise"):
            try:
                mesh = add_noise(st.session_state.original_mesh_aug.copy(), noise_level)
                st.session_state.noisy_mesh = mesh
                st.success("✅ Noise added.")
            except Exception as e:
                st.error(f"Error adding noise: {e}")
        if st.session_state.noisy_mesh is not None:
            fig = plot_mesh(st.session_state.noisy_mesh, title="Noisy Mesh")
            st.plotly_chart(fig)
            download_mesh_button(st.session_state.noisy_mesh, "Noisy Mesh", "noisy_mesh.obj")

def download_mesh_button(mesh, technique_name, file_name):
    """Provide a download button for the mesh."""
    buffer = mesh.export(file_type='obj')
    st.download_button(
        label=f"📥 Download {technique_name}",
        data=buffer,
        file_name=file_name,
        mime='text/plain'
    )
