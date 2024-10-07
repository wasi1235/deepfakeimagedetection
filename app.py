import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('./my_model.h5')

# Function to preprocess image
def preprocess_image(image):
    img = np.array(image.convert('L'))  # Convert to grayscale
    img = cv2.resize(img, (128, 128))   # Resize to model input size
    img = img.reshape((-1, 128, 128, 1))  # Reshape to fit model input
    img = img.astype('float32') / 255.0  # Normalize
    return img

# Function to make prediction
def predict_image(image):
    img = preprocess_image(image)
    prediction = model.predict(img)
    prediction = np.round(prediction)  # Assuming binary output (0 or 1)
    return "Fake" if prediction == 0 else "Real"

# Streamlit app interface
st.title("Real or Fake Face Detection")

# File uploader for user to upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Make prediction
    prediction = predict_image(image)
    
    st.write(f"Predicted Status: **{prediction}**")
