import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('./my_model.h5')

model = load_model()

# Function to preprocess the image
def preprocess_image(image):
    img = np.array(image.convert('L'))  # Convert to grayscale
    img = cv2.resize(img, (128, 128))   # Resize to model input size
    img = img.reshape((-1, 128, 128, 1))  # Reshape to fit model input
    img = img.astype('float32') / 255.0  # Normalize to [0, 1]
    return img

# Function to make prediction
def predict_image(image):
    try:
        img = preprocess_image(image)
        prediction = model.predict(img)
        prediction = np.round(prediction)  # Assuming binary output (0 or 1)
        return "Fake" if prediction == 0 else "Real"
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return "Error"

# Streamlit app interface
st.title("Real or Fake Face Detection")

# File uploader for user to upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Make prediction
        prediction = predict_image(image)
        
        if prediction != "Error":
            st.write(f"Predicted Status: **{prediction}**")
    except Exception as e:
        st.error(f"Error processing the image: {e}")
