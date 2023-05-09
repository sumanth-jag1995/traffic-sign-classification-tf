import pickle
import cv2
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

# Load the model
model = pickle.load(open("artifacts/model.pkl", "rb"))

def predict_sign_class(img, model, sign_classes):
    # Resize the image and convert to numpy array
    image_np = np.array(cv2.resize(img, (32, 32), interpolation = cv2.INTER_AREA))
    
    # Convert image to greyscale
    img_gry = np.sum(image_np/3, axis=2, keepdims=True)
    
    # Normalize the image
    img_norm_gry = (img_gry-128)/128

    # Get prediction probabilities array 
    pred_prob = model.predict(np.array([img_norm_gry]))
    
    # Get highest prediction probability & match it to traffic sign class names list
    pred_class = sign_classes['Name'][pred_prob.argmax()] 
    print(pred_class)

    return pred_class, pred_prob.max()

def main():
    st.title("Traffic Sign Classification")
    
    # Read the traffic sign classes data and load into DataFrame
    sign_classes = pd.read_csv("artifacts/data/labels.csv")
    
    # Upload file for classification
    uploaded_file = st.file_uploader("Choose a traffic sign image", type="png")
    
    if uploaded_file is not None and st.button("Predict"):
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        st.image(image, caption='Uploaded Traffic Sign.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        pred_class, pred_prob = predict_sign_class(image_np, model, sign_classes)
        st.write(f"Prediction: {pred_class}")
        st.write(f"Probability: {pred_prob:.2f}")            

if __name__ == "__main__":
    main()