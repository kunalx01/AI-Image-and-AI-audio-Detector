import os
import cv2
import numpy as np
import joblib
import torch
from transformers import ViTFeatureExtractor, ViTModel
from tensorflow import keras
from keras.applications import ResNet50
from keras.models import Model
from scipy.fftpack import fft2, fftshift
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray

# Initialize models and feature extractors
def initialize_models():
    # Initialize ResNet50
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    cnn_feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)
    
    # Initialize ViT
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224', do_rescale=False)
    vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit_model = vit_model.to(device)
    
    return cnn_feature_extractor, feature_extractor, vit_model

# Feature extraction functions
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    return np.expand_dims(image, 0)

def extract_cnn_features(image, cnn_feature_extractor):
    cnn_features = cnn_feature_extractor.predict(image)
    return cnn_features.reshape(cnn_features.shape[0], -1)

def extract_vit_features(image, feature_extractor, vit_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = vit_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embedding

def extract_frequency_features(image):
    gray_img = cv2.cvtColor((image[0] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    f_transform = fft2(gray_img)
    f_transform_shifted = fftshift(f_transform)
    magnitude_spectrum = np.abs(f_transform_shifted)
    high_freq = magnitude_spectrum[0:30, 0:30]
    high_freq_mean = np.mean(high_freq)
    return np.array([high_freq_mean]).reshape(-1, 1)

def extract_lbp_features(image):
    radius = 1
    n_points = 8 * radius
    gray_img = rgb2gray(image[0])
    lbp = local_binary_pattern(gray_img, n_points, radius, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist / lbp_hist.sum()
    return np.array([lbp_hist])

def extract_skin_tone_features(image):
    img = (image[0] * 255).astype(np.uint8)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv_img, lower_skin, upper_skin)
    skin_region = cv2.bitwise_and(hsv_img, hsv_img, mask=skin_mask)
    skin_pixels = skin_region[skin_mask > 0]
    
    if len(skin_pixels) > 0:
        mean_hue = np.mean(skin_pixels[:, 0])
        mean_saturation = np.mean(skin_pixels[:, 1])
        mean_value = np.mean(skin_pixels[:, 2])
        std_hue = np.std(skin_pixels[:, 0])
        std_saturation = np.std(skin_pixels[:, 1])
        std_value = np.std(skin_pixels[:, 2])
        return np.array([[mean_hue, mean_saturation, mean_value, std_hue, std_saturation, std_value]])
    return np.array([[0, 0, 0, 0, 0, 0]])

def detect_image(image_path, model_path="image_classifier.pkl"):
    """
    Detect if an image is AI-generated or real.
    
    Args:
        image_path (str): Path to the image file
        model_path (str): Path to the saved model file
    
    Returns:
        tuple: (prediction_label, confidence_score)
        - prediction_label: String indicating "AI-generated" or "Real"
        - confidence_score: Float between 0 and 1 indicating confidence
    """
    try:
        # Initialize models
        cnn_feature_extractor, feature_extractor, vit_model = initialize_models()
        
        # Load the trained model
        loaded_model = joblib.load(model_path)
        
        # Preprocess image
        image = preprocess_image(image_path)
        
        # Extract features
        cnn_feat = extract_cnn_features(image, cnn_feature_extractor)
        vit_feat = extract_vit_features(image, feature_extractor, vit_model)
        freq_feat = extract_frequency_features(image)
        lbp_feat = extract_lbp_features(image)
        skin_tone_feat = extract_skin_tone_features(image)
        
        # Combine features
        combined_features = np.hstack([cnn_feat, vit_feat, freq_feat, lbp_feat, skin_tone_feat])
        
        # Make prediction
        prediction = loaded_model.predict(combined_features)
        prediction_proba = loaded_model.decision_function(combined_features)
        confidence_score = abs(prediction_proba[0])  # Using absolute value of decision function as confidence
        
        # Get prediction label
        label = "AI-generated" if prediction[0] == 1 else "Real"
        
        # Print result
        print(f"The image is predicted to be: {label}")
        print(f"Confidence Score: {confidence_score:.2f}")
        
        return label, confidence_score
        
    except Exception as e:
        print(f"Error during detection: {str(e)}")
        return None, None

if __name__ == "__main__":
    # Example usage
    image_path = "testImage.jpg"  # Replace with your image path
    result = detect_image(image_path)