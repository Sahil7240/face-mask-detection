
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("face_mask_model.keras")

st.title("😷 Face Mask Detection System")
st.write("Upload an image to predict whether a person is wearing a face mask.")

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

        prediction = float(
            model.predict(image_array, verbose=0)[0][0]
        )

        if prediction >= 0.5:
            st.error("Prediction: WITHOUT MASK")
        else:
            st.success("Prediction: WITH MASK")

        st.caption(
            "This is an image classifier, not a face-localization system. "
            "Predictions may be incorrect."
        )
