import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import pickle
from skimage import feature, color
import base64


# ----------------- Page Configuration -----------------
st.set_page_config(
    page_title="Brain Tumor Detection System",
    page_icon="🧠",
    layout="centered"
)

# ----------------- Load Model and Labels -----------------
@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model("brain_tumor_model.h5")
    with open("class_labels.pkl", "rb") as f:
        class_labels = pickle.load(f)
    return model, class_labels

model, class_labels = load_model_and_labels()

# ---------Helper Functions for Checking if the Uploaded Image Looks Like a Real MRI --------------
def is_valid_mri(image: Image.Image) -> bool:
    """Heuristically check if the uploaded image is likely a brain MRI."""
    img = np.array(image)
    gray = color.rgb2gray(img) if len(img.shape) == 3 else img

    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    edges = feature.canny(gray, sigma=2)
    edge_density = np.mean(edges)

    if mean_intensity > 0.9 or mean_intensity < 0.05:
        return False
    if std_intensity < 0.02 or std_intensity > 0.35:
        return False
    if edge_density > 0.15:
        return False
    return True

#-----Preprocessing Image for CNN Input------
def preprocess_image(image):
    """Resize and normalize image for model input."""
    input_shape = model.input_shape[1:3]
    image = image.resize(input_shape)
    img_array = np.array(image) / 255.0

    # Handle different image modes safely
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[-1] == 4:  # RGBA → RGB
        img_array = img_array[..., :3]
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ----------------- UI Layout -----------------
st.title("🧠 Brain Tumor Detection System")
st.markdown("Upload a **Brain MRI Image** to predict the tumor type using a trained CNN model.")

if "reset" not in st.session_state:
    st.session_state.reset = False

if st.session_state.reset:
    uploaded_file = None
    st.session_state.reset = False
else:
    uploaded_file = st.file_uploader(
        "Upload MRI Image",
        type=["jpg", "jpeg", "png"],
        help="Upload a valid brain MRI scan image (JPG, JPEG, PNG formats only)."
    )

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)

        # Validate MRI image quality
        if not is_valid_mri(image):
            st.error("⚠️ The uploaded image doesn't appear to be a valid brain MRI. Please try another image.")
        else:
            # Center the uploaded MRI image
            img_bytes = uploaded_file.getvalue()
            encoded = base64.b64encode(img_bytes).decode()

            st.markdown(
                f"""
                <div style="text-align:center;">
                    <img src="data:image/png;base64,{encoded}" width="200">
                    <p><i>🧩 Uploaded MRI Image</i></p>
                </div>
                """,
                unsafe_allow_html=True
                )
            
            if st.button("🔍 Predict Tumor Type"):
                with st.spinner("Analyzing MRI image..."):
                    processed_img = preprocess_image(image)
                    prediction = model.predict(processed_img)[0]
                    sorted_indices = np.argsort(prediction)[::-1]
                    sorted_labels = [class_labels[i] for i in sorted_indices]
                    sorted_probs = [prediction[i] for i in sorted_indices]

                st.success(f"🩺 **Predicted Tumor Type:** {sorted_labels[0]}")

                st.subheader("📊 Prediction Probabilities")
                for label, prob in zip(sorted_labels, sorted_probs):
                    st.write(f"**{label}** — {prob*100:.2f}%")
                    st.progress(float(prob))

                if st.button("🖼️ Try Another Image"):
                    st.session_state.reset = True
                    st.rerun()

    except Exception as e:
        # Simplified error message for non-technical users
        if "size 3 along channel_axis" in str(e) or "RGBA" in str(e):
            st.error("⚠️ The uploaded image format isn’t supported. Please upload a proper brain MRI image (JPG or PNG).")
        else:
            st.error("⚠️ Could not process the uploaded image. Please make sure it is a clear MRI scan of the brain.")

# ----------------- Footer -----------------
st.markdown("---")
st.markdown("Developed by **Binod Thapa** | Brain Tumor Detection using CNN 🧠")
