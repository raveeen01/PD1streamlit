import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the trained model (ensure that 'mnist_vgg16_model.h5' is in the same directory as this file)
model = load_model('SSD_VGG16_Mode.h5')

# Define a function to preprocess the image
def preprocess_image(uploaded_image):
    img = Image.open(uploaded_image).convert('L')  # Convert to grayscale
    img = img.resize((32, 32))                    # Resize to match VGG16 input size
    img = img.convert('RGB')                      # Convert grayscale to RGB
    img = np.array(img) / 255.0                   # Normalize pixel values
    img = np.expand_dims(img, axis=0)             # Add batch dimension
    return img

# Streamlit app title and description
st.title("MNIST Digit Classifier")
st.write("Upload a handwritten digit image, and the model will classify it.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    img = preprocess_image(uploaded_file)
    
    # Make prediction
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    # Display the prediction
    st.write(f"Predicted Class: {predicted_class}")
