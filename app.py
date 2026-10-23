
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Face Mask Detection", page_icon="😷")
st.title("Face Mask Detection System")
st.write("Upload a clear, front-facing photo.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("face_mask_model.keras")

@st.cache_resource
def load_face_detector():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

uploaded_file = st.file_uploader(
    "Upload a face image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    if st.button("Predict"):
        img = np.array(image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        detector = load_face_detector()
        faces = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces) == 0:
            st.warning(
                "No face detected. Try a clear, front-facing photo "
                "with the face closer to the camera."
            )
        else:
            # Select the largest detected face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

            # Add a small margin around the face
            margin = int(0.15 * max(w, h))
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(img.shape[1], x + w + margin)
            y2 = min(img.shape[0], y + h + margin)

            face = img[y1:y2, x1:x2]
            face = cv2.resize(face, (64, 64))

            model = load_model()
            batch = np.expand_dims(face.astype(np.uint8), axis=0)
            score = float(model.predict(batch, verbose=0)[0][0])

            st.image(face, caption="Face used for prediction", width=200)

            if score >= 0.5:
                st.error(f"WITHOUT MASK — score: {score:.1%}")
            else:
                st.success(f"WITH MASK — score: {1-score:.1%}")

            st.caption(
                "Experimental classifier. Predictions can be incorrect, "
                "especially for side-facing faces or unusual masks."
            )
