import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

# Load model
model = load_model("digit_model.h5")

st.title("AI-Based Handwritten Digit Recognition")

uploaded_file = st.file_uploader(
    "Upload a handwritten digit image",
    type=["png","jpg","jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", width=200)

    # Convert to grayscale
    img = image.convert("L")

    # Resize
    img = img.resize((28,28))

    # Invert colors (important for MNIST)
    img = ImageOps.invert(img)

    # Convert to array
    img = np.array(img)

    # Normalize
    img = img.astype("float32") / 255.0

    # Reshape
    img = img.reshape(1,28,28)

    # Predict
    prediction = model.predict(img)

    digit = np.argmax(prediction)

    confidence = np.max(prediction)*100

    st.success(f"Predicted Digit : {digit}")

    st.write(f"Confidence : {confidence:.2f}%")