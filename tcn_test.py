import numpy as np
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch.nn as nn

# Load Wav2Vec model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

# Model parameters
combined_feature_size = 798
output_size = 2
num_channels = [64, 128, 256]

class TCNModel(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=3, dropout=0.2):
        super(TCNModel, self).__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(input_size, num_channels[0], kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels[0], num_channels[1], kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels[1], num_channels[2], kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc = nn.Linear(num_channels[2], output_size)
    
    def forward(self, x):
        x = self.tcn(x)
        x = torch.mean(x, dim=2)
        output = self.fc(x)
        return output

# Instantiate the model and load the saved parameters
model = TCNModel(input_size=combined_feature_size, output_size=output_size, num_channels=num_channels)
model.load_state_dict(torch.load('tcn_New_model.pth'))
model.eval()

def extract_traditional_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    traditional_features = np.vstack([mfccs, chroma, spectral_centroid, spectral_bandwidth, spectral_rolloff, zero_crossing_rate, rms])
    
    # Print each feature's name and values
    #print("Chroma:", chroma)
    #print("Spectral Centroid:", spectral_centroid)
    #print("Spectral Bandwidth:", spectral_bandwidth)
    #print("Zero Crossing Rate:", zero_crossing_rate)
    #print("Spectral Rolloff:", spectral_rolloff)
    #print("RMS:", rms)
    
    return traditional_features.T 

def extract_wav2vec_features(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values
    with torch.no_grad():
        embeddings = wav2vec_model(input_values).last_hidden_state
    # Print Wav2Vec embeddings
    #print("Wav2Vec Embeddings:", embeddings.squeeze(0).numpy())
    
    return embeddings.squeeze(0).numpy()


def format_audio_features(audio_file_path):
    # Extract traditional and wav2vec features
    traditional_features = extract_traditional_features(audio_file_path)
    wav2vec_features = extract_wav2vec_features(audio_file_path)
    
    # Combine traditional and wav2vec features for prediction
    combined_features = extract_combined_features(audio_file_path)
    
    # Model prediction
    prediction_result = detect_ai_audio_tcn(audio_file_path)
    
    # Summarize MFCC values (mean and std dev of first 13 coefficients, limited to first 10 values)
    mfccs = np.array(traditional_features[:13])[:10]  # Limit to first 10 values
    mfcc_summary = {
        "mean": mfccs.mean(axis=0).tolist(),
        "std_dev": mfccs.std(axis=0).tolist()
    }
    
    # Summarize Chroma features (limited to first 10 values)
    chroma = np.array(traditional_features[13:25])[:10]  # Limit to first 10 values
    chroma_summary = {
        "mean": chroma.mean(axis=0).tolist(),
        "max": chroma.max(axis=0).tolist(),
        "min": chroma.min(axis=0).tolist()
    }
    
    # Organize features into the requested format, limiting to the first 10 values where applicable
    formatted_data = {
        "MFCCs": mfcc_summary,  # MFCC summary
        "Chroma": chroma_summary,  # Chroma summary
        #"Spectral Centroid": np.mean(traditional_features[25:26][:10]).tolist(),  # First 10 values
        #"Spectral Bandwidth": np.mean(traditional_features[26:27][:10]).tolist(),  # First 10 values
        #"Spectral Rolloff": np.mean(traditional_features[27:28][:10]).tolist(),  # First 10 values
        #"Zero Crossing Rate": np.mean(traditional_features[28:29][:10]).tolist(),  # First 10 values
        #"RMS": np.mean(traditional_features[29:30][:10]).tolist(),  # First 10 values
        "Wav2Vec Embeddings": wav2vec_features[:10].tolist(),  # Limit Wav2Vec to first 10 values
        "Combined result": {
            "Is AI generated": prediction_result["is_ai_generated"],
            "Confidence": prediction_result["confidence"]
        },
        "CNN Model result": {
            "Is AI generated": False,  # Example value; replace with actual CNN model result
            "Confidence": 0.91  # Example confidence; replace with actual value
        },
        "TCN Model result": {
            "Is AI generated": prediction_result["is_ai_generated"],
            "Confidence": prediction_result["confidence"]
        }
    }
    
    return formatted_data



# Example usage:
# formatted_features = format_audio_features("path/to/your/audio/file.wav")
# print(formatted_features)


def extract_combined_features(file_path):
    traditional_features = extract_traditional_features(file_path)
    wav2vec_features = extract_wav2vec_features(file_path)
    min_len = min(traditional_features.shape[0], wav2vec_features.shape[0])
    combined_features = np.concatenate((traditional_features[:min_len], wav2vec_features[:min_len]), axis=1)
    return combined_features.T 

def detect_ai_audio_tcn(audio_file_path):
    features = extract_combined_features(audio_file_path)
    # print(features)
    
    target_length = 100
    if features.shape[1] < target_length:
        features = np.pad(features, ((0, 0), (0, target_length - features.shape[1])), mode='constant')
    else:
        features = features[:, :target_length]
    
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        output = model(features_tensor)
        prediction = torch.softmax(output, dim=1)
        is_ai_generated = prediction[0, 1].item() > 0.5
        confidence = prediction[0, 1].item() if is_ai_generated else prediction[0, 0].item()
    
    return {
        "is_ai_generated": is_ai_generated,
        "confidence": float(confidence),
        "prediction_value": float(prediction[0, 1].item())
    }

# Example usage:
# result = detect_ai_audio_tcn("path/to/your/audio/file.wav")
# print(result) 