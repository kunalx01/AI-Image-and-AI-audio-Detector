import cv2
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import spectrogram
from tensorflow import keras 
from keras.models import load_model
import os
import tempfile

# Load the saved model
loaded_model = load_model("cnn_model.h5")

def preprocess_spectrogram(img_path, img_size=(640, 480)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size)
    img = img / 255.0
    img = img.reshape(1, img_size[0], img_size[1], 1)
    return img

def create_spectrogram(audio_path, output_path):
    audio, sample_rate = sf.read(audio_path)
    
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    f, t, Sxx = spectrogram(audio, sample_rate)
    
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='auto')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Intensity [dB]')
    plt.title('Spectrogram')
    plt.savefig(output_path)
    plt.close()

def detect_ai_audio(audio_file_path):
    # Create a temporary directory to store the spectrogram
    with tempfile.TemporaryDirectory() as temp_dir:
        spectrogram_path = os.path.join(temp_dir, "temp_spectrogram.png")
        
        # Generate spectrogram
        create_spectrogram(audio_file_path, spectrogram_path)
        
        # Preprocess the spectrogram
        preprocessed_image = preprocess_spectrogram(spectrogram_path)
        
        # Make prediction
        prediction = loaded_model.predict(preprocessed_image)[0][0]
        
        # Interpret the result
        is_ai_generated = prediction >= 0.80
        confidence = prediction if is_ai_generated else 1 - prediction
        
        return {
            "is_ai_generated": is_ai_generated,
            "confidence": float(confidence),
            "prediction_value": float(prediction)
        }

# Example usage:
# result = detect_ai_audio("path/to/your/audio/file.wav")
# print(result)