
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷"
)

st.title("Face Mask Detection System")
st.write("Upload a clear image of a person to predict mask status.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("face_mask_model.keras")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    if st.button("Predict"):
        model = load_model()

        image_array = np.array(
            image.resize((64, 64)),
            dtype=np.uint8
        )
        image_array = np.expand_dims(image_array, axis=0)

        score = float(
            model.predict(image_array, verbose=0)[0][0]
        )

        st.write(f"Raw model score: {score:.4f}")

        if score >= 0.5:
            st.error("Prediction: WITHOUT MASK")
            st.write(f"Model score: {score:.1%}")
        else:
            st.success("Prediction: WITH MASK")
            st.write(f"Model score: {1-score:.1%}")

st.caption(
    "Note: This classifier evaluates the entire image. "
    "It may give incorrect results for unclear images or images without faces."
)
