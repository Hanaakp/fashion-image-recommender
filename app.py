import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

from tensorflow.keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors
import os

# Load precomputed data
features = np.load('features.npy')
image_paths = np.load('paths.npy', allow_pickle=True)

# Load model
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Set up KNN
nn = NearestNeighbors(n_neighbors=6, metric='cosine')
nn.fit(features)

# Helper: process image
def extract_feature(img):
    img = img.convert('RGB').resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# UI
st.title("Fashion Recommendation System")

uploaded_file = st.file_uploader("Upload a fashion image", type=['jpg', 'png'])

if uploaded_file:
    img = Image.open(uploaded_file)
    

    # Feature extraction
    st.image(img, caption='Uploaded Image', use_container_width=True)

    query_feat = model.predict(extract_feature(img), verbose=0)
    distances, indices = nn.kneighbors(query_feat)

    st.subheader("Similar Fashion Items:")
    cols = st.columns(5)
    for i, idx in enumerate(indices[0][1:]):  # Skip self
        with cols[i]:
            st.image(image_paths[idx], use_container_width=True)
