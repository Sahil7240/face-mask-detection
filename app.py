
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
            st.error(f"Prediction: WITHOUT MASK ({score:.1%})")
        else:
            st.success(f"Prediction: WITH MASK ({1-score:.1%})")

        st.caption(
            "This model classifies the whole uploaded image. "
            "It may misclassify images without a clear face."
        )
