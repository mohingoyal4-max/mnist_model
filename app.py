import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import os

# 1. Setup & Integration
@st.cache_resource
def load_trained_model():
    """
    Loads the pre-trained model. We use st.cache_resource to load it once,
    preventing retraining or reloading upon every user interaction.
    """
    model_path = 'mnist_model.h5'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

def extract_layer_features(model, input_data):
    """
    Extracts the layer-wise features (activations) for the uploaded image 
    as it passes through the dense layers.
    """
    # Isolate only the dense layers to visualize their activations
    dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
    layer_outputs = [layer.output for layer in dense_layers]
    
    # Create a feature extractor model
    feature_extractor = tf.keras.models.Model(inputs=model.inputs, outputs=layer_outputs)
    
    # Get the activations
    activations = feature_extractor.predict(input_data)
    
    # Return activations mapped to layer names
    return list(zip([layer.name for layer in dense_layers], activations))

def preprocess_image(image):
    """
    Preprocess the uploaded image to match the MNIST format:
    28x28 pixels, grayscale, flattened (1D array of 784), normalized (0-1).
    """
    # Convert to grayscale
    gray_image = ImageOps.grayscale(image)
    
    # Invert the image (MNIST dataset has white digits on black background).
    # Most user uploads are dark digits on bright background.
    # We dynamically check the top-left corner pixel to decide if inversion is needed.
    img_array = np.array(gray_image)
    if img_array[0, 0] > 127: # If background appears bright
        gray_image = ImageOps.invert(gray_image)
        
    # Resize to 28x28
    resized_image = gray_image.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to NumPy array, normalize, and flatten
    normalized_array = np.array(resized_image).astype('float32') / 255.0
    flattened_array = normalized_array.reshape(1, 784)
    
    return flattened_array, resized_image

def main():
    # 2. User Interface
    st.set_page_config(page_title="MNIST Digit Classifier", layout="wide")
    st.title("MNIST Digit Classifier")
    st.markdown("Upload a digit image to see the model prediction and its dense layer activations.")
    
    # Load Model
    model = load_trained_model()
    if model is None:
        st.error("Pre-trained model (`mnist_model.h5`) not found. Please run `train.py` first to generate it.")
        return
        
    st.sidebar.success("Pre-trained model loaded successfully!")
    
    # Include a file uploader ("Browse") that accepts image files
    uploaded_file = st.file_uploader("Browse", type=['jpg', 'jpeg', 'png'])
    
    # 3. Prediction & Display
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Split layout for the Image and the Prediction Output
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, caption="Original Input", use_container_width=True)
            
            # Preprocess the image
            flattened_input, processed_img_display = preprocess_image(image)
            st.image(processed_img_display, caption="Processed Image (28x28)", width=150)
            
        with col2:
            st.subheader("Prediction")
            # Predict the digit using the loaded model
            predictions = model.predict(flattened_input)
            predicted_digit = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            # Display prominent prediction
            st.markdown(f"<h1 style='text-align: center; font-size: 100px; color: #4CAF50;'>{predicted_digit}</h1>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align: center;'>Confidence: {confidence:.2%}</h3>", unsafe_allow_html=True)
            
        # Displaying Layer-wise Features
        st.divider()
        st.subheader("Layer-wise Features (Activations)")
        st.markdown("Visual representation of the activations inside the Dense layers for the uploaded image:")
        
        layer_activations = extract_layer_features(model, flattened_input)
        
        # Display horizontal bar charts for layer activations
        for layer_name, activation in layer_activations:
            st.markdown(f"**Layer: `{layer_name}`** (Shape: {activation.shape})")
            # We use st.bar_chart. Activation array is 2D -> 1xUnits. We plot the 1D Array.
            st.bar_chart(activation[0])
            
    # Display the Evaluation Artifacts Below the Prediction area
    st.divider()
    st.subheader("Model Evaluation Artifacts (from Training)")
    
    col3, col4 = st.columns(2)
    
    with col3:
        if os.path.exists('loss_plot.png'):
            st.image('loss_plot.png', caption="Loss vs. Epoch", use_container_width=True)
        else:
            st.info("Performance plot `loss_plot.png` is not generated yet.")
            
    with col4:
        if os.path.exists('confusion_matrix.png'):
            st.image('confusion_matrix.png', caption="Confusion Matrix", use_container_width=True)
        else:
            st.info("Confusion matrix `confusion_matrix.png` is not generated yet.")

if __name__ == "__main__":
    main()
