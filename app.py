import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

from src.metrics import evaluate
from src.cartoonize import cartoonize_image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="IMAGE CARTOONIZER",
    page_icon="🎨",
    layout="centered"
)

# ---------------- DARK THEME ----------------
st.markdown("""
<style>
/* Background */
.stApp {
    background-color: #0f172a;
}

/* Main container */
.block-container {
    padding-top: 2rem;
    max-width: 900px;
}

/* Title */
h1 {
    text-align: center;
    color: #ffffff !important;
    font-weight: 800;
    font-size: 42px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 25px;
}

/* Labels */
label {
    color: #cbd5f5 !important;
}

/* Image captions */
.stCaption {
    color: #e2e8f0 !important;
    font-size: 14px;
    font-weight: 500;
}

/* Metrics text */
.metrics {
    color: #f1f5f9;
    font-size: 16px;
    margin-top: 10px;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: #1e293b;
    padding: 12px;
    border-radius: 12px;
}

/* Select box */
[data-baseweb="select"] {
    background-color: #1e293b;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #7c3aed, #3b82f6);
    color: white;
    border-radius: 10px;
    height: 48px;
    font-size: 16px;
    font-weight: 600;
}

/* Download button */
.stDownloadButton>button {
    border-radius: 10px;
}

/* Success box */
.stAlert {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<h1>Image Cartoonizer</h1>", unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image and convert it into a cartoon</div>', unsafe_allow_html=True)

# ---------------- INPUT ----------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
method = st.selectbox("Method", ["kmeans", "meanshift"])

st.divider()

os.makedirs("output_images", exist_ok=True)

# ---------------- PROCESS ----------------
if uploaded_file:
    image = Image.open(uploaded_file)
    img_np = np.array(image)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<p style='color:#e2e8f0;'>Original Image</p>", unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    with col2:
        st.markdown("<p style='color:#e2e8f0;'>Cartoon Output</p>", unsafe_allow_html=True)
        output_placeholder = st.empty()

    if st.button("Cartoonize", use_container_width=True):
        with st.spinner("Processing..."):
            output = cartoonize_image(img_cv, method)

            output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            output_placeholder.image(output_rgb, use_container_width=True)

            path = f"output_images/cartoon_{method}.jpg"
            cv2.imwrite(path, output)

            st.success("Saved successfully!")

            # ---------------- METRICS ----------------
            ssim_score, mse_score = evaluate(img_cv, output)

            st.markdown("<h3 style='color:#ffffff;'>Evaluation Metrics</h3>", unsafe_allow_html=True)

            st.markdown(
                f"""
                <div class="metrics">
                    <b>SSIM:</b> {ssim_score:.4f} <br>
                    <b>MSE:</b> {mse_score:.2f}
                </div>
                """,
                unsafe_allow_html=True
            )

            # ---------------- DOWNLOAD ----------------
            _, buffer = cv2.imencode(".jpg", output)

            st.download_button(
                label="Download Image",
                data=buffer.tobytes(),
                file_name="cartoon.jpg",
                mime="image/jpeg",
                use_container_width=True
            )