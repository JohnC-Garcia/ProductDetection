import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import numpy as np
import zipfile
import io
import os

st.set_page_config(page_title="Retail Vegetable Detection Demo")
st.title("Product Detection on Retail Store Shelves")
st.write("This demo simulates detection only for the first few frames if the file uploaded is a video.")

@st.cache_resource
def load_model():
    return YOLO("yolo11n_best.pt")

model = load_model()

def draw_frame_number_rgb(image_rgb, frame_idx):
    image_with_text = image_rgb.copy()
    image_bgr = cv2.cvtColor(image_with_text, cv2.COLOR_RGB2BGR)
    cv2.putText(image_bgr, f"Frame {frame_idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4"])
if uploaded_file:
    if uploaded_file.type.startswith("image"):
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image")
        with st.spinner("Detecting..."):
            result = model(img)
            st.image(result[0].plot(), caption="Detected Vegetables")

    elif uploaded_file.type == "video/mp4":
        st.video(uploaded_file)
        st.write("Processing video (only 5 frames)...")

        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        frame_count = 0
        preview_pairs = []
        annotated_images = []

        while cap.isOpened() and len(preview_pairs) < 5:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % 10 == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb) 
                result = model(frame_pil)

                # Convert plot to RGB for correct display
                annotated_bgr = result[0].plot()
                annotated = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

                orig_with_number = draw_frame_number_rgb(frame_rgb, frame_count)
                annotated_with_number = draw_frame_number_rgb(annotated, frame_count)

                preview_pairs.append((orig_with_number, annotated_with_number))
                annotated_images.append(Image.fromarray(annotated_with_number))
            frame_count += 1

        cap.release()

        st.subheader("🔍 Detection Preview: Original vs. Labeled")
        for i, (original, labeled) in enumerate(preview_pairs):
            st.markdown(f"**Frame {i+1}**")
            col1, col2 = st.columns(2)
            with col1:
                st.image(original, caption="Original", use_container_width=True)
            with col2:
                st.image(labeled, caption="With Labels", use_container_width=True)

        # Create a ZIP file in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
            for idx, image in enumerate(annotated_images):
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                zip_file.writestr(f"annotated_frame_{idx + 1}.png", img_byte_arr.getvalue())

        st.download_button(
            label="📥 Download Annotated Frames (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="annotated_frames.zip",
            mime="application/zip"
        )
