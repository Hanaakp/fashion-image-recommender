# Fashion Image Recommender 

This project is an unsupervised fashion recommendation system that suggests visually similar fashion items based on an uploaded image. It uses a pre-trained ResNet50 model to extract visual features from images and applies k-Nearest Neighbors (KNN) to find and display similar items.

## Features
- Upload a clothing/fashion image
- See top 5 similar fashion images
- Built with TensorFlow, scikit-learn, and Streamlit
- Uses ResNet50 for feature extraction

##  Tech Stack
- Python
- TensorFlow / Keras
- scikit-learn
- NumPy
- PIL
- Streamlit

##  How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/your-username/fashion-image-recommender.git
   cd fashion-image-recommender
## ⚠️ Note
Before running the app, you need to extract the image features:

```bash
python feature_extraction.py
