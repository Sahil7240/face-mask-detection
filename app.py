
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Face Mask Detection System",
    page_icon="😷",
    layout="centered"
)

st.title("😷 Face Mask Detection System")
st.write(
    "Upload a clear, front-facing image of a person "
    "to check whether a face mask is being worn."
)

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "face_mask_model.keras",
        compile=False
    )

# Load OpenCV face detector safely
@st.cache_resource
def load_face_detector():
    cascade_path = os.path.join(
        cv2.data.haarcascades,
        "haarcascade_frontalface_default.xml"
    )

    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        raise RuntimeError(
            "Face detector could not load. "
            "Please check the OpenCV installation."
        )

    return detector


# Upload image
uploaded_file = st.file_uploader(
    "Choose a face image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        img = np.array(image)

        st.image(
            image,
            caption="Uploaded Image",
            width=350
        )

        if st.button("Detect Face Mask", type="primary"):

            with st.spinner("Analyzing image..."):

                # Detect faces
                detector = load_face_detector()

                gray = cv2.cvtColor(
                    img, cv2.COLOR_RGB2GRAY
                )

                faces = detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(40, 40)
                )

                if len(faces) == 0:
                    st.warning(
                        "No face detected. Please upload "
                        "a clear, front-facing photo."
                    )

                else:
                    # Choose the largest detected face
                    x, y, w, h = max(
                        faces,
                        key=lambda face: face[2] * face[3]
                    )

                    # Crop the detected face with a margin
                    margin = int(0.15 * max(w, h))

                    x1 = max(0, x - margin)
                    y1 = max(0, y - margin)
                    x2 = min(img.shape[1], x + w + margin)
                    y2 = min(img.shape[0], y + h + margin)

                    face = img[y1:y2, x1:x2]

                    # Prepare image for model
                    face_resized = cv2.resize(
                        face, (64, 64)
                    )

                    model_input = np.expand_dims(
                        face_resized.astype(np.float32),
                        axis=0
                    )

                    # Predict
                    model = load_model()

                    score = float(
                        model.predict(
                            model_input, verbose=0
                        )[0][0]
                    )

                    st.image(
                        face,
                        caption="Detected face used for prediction",
                        width=220
                    )

                    # Model label 0 = With Mask
                    # Model label 1 = Without Mask
                    if score >= 0.5:
                        st.error("Prediction: WITHOUT MASK")
                        st.write(
                            f"Without-mask score: {score:.1%}"
                        )
                    else:
                        st.success("Prediction: WITH MASK")
                        st.write(
                            f"With-mask score: {(1 - score):.1%}"
                        )

                    st.caption(
                        "This is an educational prototype. "
                        "Predictions may be incorrect, especially "
                        "for side views, unusual masks, or poor lighting."
                    )

    except Exception as e:
        st.error("Unable to process the image.")
        st.write("Error details:", str(e))

st.divider()
st.caption(
    "Machine Learning ISE-2 | Face Mask Detection"
)
