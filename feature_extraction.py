# feature_extraction.py

import os
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

from tensorflow.keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors

# image_dir = 'images'
base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(base_dir, 'images')
image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.lower().endswith(('.jpg', '.png'))]

model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def load_and_preprocess(path):
    img = Image.open(path).convert('RGB').resize((224, 224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return preprocess_input(arr)

features = []
valid_paths = []

for path in image_paths:
    try:
        img_tensor = load_and_preprocess(path)
        feat = model.predict(img_tensor, verbose=0)
        features.append(feat.flatten())
        valid_paths.append(path)
    except:
        continue

np.save('features.npy', np.array(features))
np.save('paths.npy', np.array(valid_paths))
