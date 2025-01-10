import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np

# Load the pre-trained model
MODEL_URL = "https://tfhub.dev/emmarex/plant-disease/1"
model = hub.load(MODEL_URL)

# Class names (example: replace with actual class names from the model documentation)
CLASS_NAMES = [
    "Apple Scab", "Apple Black Rot", "Apple Healthy", 
    "Corn Gray Leaf Spot", "Corn Common Rust", "Corn Healthy",
    "Grape Black Rot", "Grape Healthy", "Potato Early Blight", 
    "Potato Late Blight", "Potato Healthy"
]

# Streamlit App
st.title("Plant Disease Detector")
st.write("Upload an image of a plant leaf to predict its health condition.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def predict_image(img):
    # Resize and preprocess image
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make predictions
    predictions = model(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)
    
    return CLASS_NAMES[predicted_class], confidence

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Predict and show result
    with st.spinner("Analyzing..."):
        label, confidence = predict_image(image)
    st.success(f"Prediction: {label} (Confidence: {confidence:.2f})")
