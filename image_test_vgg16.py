import tensorflow as tf
import numpy as np
import os

def load_and_preprocess_image(image_path):
    """
    Load and preprocess a single image for prediction
    Returns:
        numpy array: Preprocessed image array or None if error occurs
    """
    try:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        return None

def predict_image(model, image_array):
    """
    Make prediction on preprocessed image
    Returns:
        tuple: (label, confidence) or (None, None) if error occurs
    """
    try:
        prediction = model.predict(image_array)[0][0]
        label = "AI-generated" if prediction >= 0.5 else "Real"
        confidence = prediction if prediction >= 0.5 else 1 - prediction
        return label, confidence * 100
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        return None, None

def analyze_image(model_path, image_path):
    """
    Analyze a single image using the trained model
    Returns:
        dict: Dictionary containing the results, including:
            - success: Boolean indicating if analysis was successful
            - error: Error message if success is False
            - filename: Name of the analyzed image
            - label: Predicted label (AI-generated or Real)
            - confidence: Confidence score in percentage
            - prediction_value: Raw prediction value
    """
    result = {
        'success': False,
        'error': None,
        'filename': os.path.basename(image_path),
        'label': None,
        'confidence': None,
        'prediction_value': None
    }
    
    try:
        # Load the model
        model = tf.keras.models.load_model(model_path)
        
        # Load and preprocess image
        img_array = load_and_preprocess_image(image_path)
        if img_array is None:
            result['error'] = "Failed to load or preprocess image"
            return result
        
        # Make prediction
        prediction = model.predict(img_array)[0][0]
        
        # Process results
        result['success'] = True
        result['label'] = "AI-generated" if prediction >= 0.5 else "Real"
        result['prediction_value'] = float(prediction)
        result['confidence'] = float(prediction if prediction >= 0.5 else 1 - prediction) * 100
        
    except Exception as e:
        result['error'] = str(e)
    
    return result

def analyze_folder(model_path, folder_path):
    """
    Analyze all images in a folder using the trained model
    Returns:
        list: List of dictionaries containing results for each image
    """
    results = []
    
    try:
        # Load the model once for all images
        model = tf.keras.models.load_model(model_path)
        
        # Process each image in the folder
        for image_file in os.listdir(folder_path):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, image_file)
                result = analyze_image(model_path, image_path)
                results.append(result)
                
    except Exception as e:
        results.append({
            'success': False,
            'error': str(e),
            'filename': 'folder_analysis_error',
            'label': None,
            'confidence': None,
            'prediction_value': None
        })
    
    return results

if __name__ == "__main__":
    # Example usage
    MODEL_PATH = 'Image_part.h5'
    
    # Example for single image
    IMAGE_PATH = 'test_image.jpg'
    result = analyze_image(MODEL_PATH, IMAGE_PATH)
    
    # Print results
    if result['success']:
        print(f"\nResults for image: {result['filename']}")
        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print(f"Raw prediction value: {result['prediction_value']:.4f}")
    else:
        print(f"Error analyzing image: {result['error']}")
    
    # Example for folder analysis
    """
    FOLDER_PATH = 'test_images'
    results = analyze_folder(MODEL_PATH, FOLDER_PATH)
    
    # Print results for each image
    for result in results:
        if result['success']:
            print(f"\nResults for image: {result['filename']}")
            print(f"Prediction: {result['label']}")
            print(f"Confidence: {result['confidence']:.2f}%")
            print(f"Raw prediction value: {result['prediction_value']:.4f}")
        else:
            print(f"Error analyzing image {result['filename']}: {result['error']}")
    """