
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷"
)

st.title("😷 Face Mask Detection System")
st.write("Upload a clear image to classify mask status.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "face_mask_model.keras",
        compile=False
    )

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=350)

    if st.button("Detect Face Mask"):
        with st.spinner("Analyzing image..."):
            model = load_model()

            image = image.resize((64, 64))
            image_array = np.array(
                image, dtype=np.float32
            )
            image_array = np.expand_dims(
                image_array, axis=0
            )

            score = float(
                model.predict(
                    image_array, verbose=0
                )[0][0]
            )

            st.write(f"Raw model score: {score:.4f}")

            if score >= 0.5:
                st.error("Prediction: WITHOUT MASK")
            else:
                st.success("Prediction: WITH MASK")

            st.caption(
                "Educational prototype. The model classifies "
                "the whole image and may give incorrect predictions."
            )
