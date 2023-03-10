import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

#1Application heading
st.title("Dermoverse Skin Cancer Detector")

#1Brief summary of what the application does
st.subheader("This BETA can classify potential skin cancer images into two classes, whether they are benign or malignant. The images uploaded should be clinically made. ")

# Define the URL you want to open when the button is clicked
url = 'https://dermoverse.org'

# Add a button to your Streamlit app
if st.button('Open website'):
    # When the button is clicked, open the URL in a new browser tab
    js = f"window.open('{url}')"  # JavaScript code to open a new tab
    html = '<img src onerror="{}">'.format(js)  # Create an invisible image that triggers the JS code
    st.write(html, unsafe_allow_html=True)  # Render the image
    
#1Information of what kind of images should be uploaded.
st.subheader("In the upcoming versions phone-made pictures will be supported.")

st.subheader("Note that is just a beta. Consult with a professional for further information.")

# Load the pre-trained model
model = tf.keras.models.load_model('dermodev.h5')

# Define the class labels
class_labels = ['Malignant', 'Benign']

# Define the Streamlit app
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# Define the "Run Model" button
if uploaded_file is not None:
    run_model = st.button("Run Model")
else:
    run_model = False

# Make a prediction on the uploaded image when the "Run Model" button is clicked
if run_model:
    image = Image.open(uploaded_file)
    image = image.resize((224, 224))
    image = np.array(image)
    image = image.astype('float32') / 255.0
    prediction = model.predict(np.expand_dims(image, axis=0))[0]
    percentages = tf.nn.softmax(prediction) * 100

    # Print the prediction
    st.write("Prediction:")
    for i, percentage in enumerate(percentages):
        class_label = class_labels[i]
        st.write(f"{class_label}: {percentage:.2f}%")


