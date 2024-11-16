import os
import time
from cnn_test import detect_ai_audio as detect_ai_audio_cnn
from tcn_test import detect_ai_audio_tcn, format_audio_features

def classify_audio(file_path):
    # Get predictions from both models
    cnn_result = detect_ai_audio_cnn(file_path)
    tcn_result = detect_ai_audio_tcn(file_path)
    features = format_audio_features(file_path)
    
    # Combine results (adjust logic as needed)
    combined_is_ai = cnn_result['is_ai_generated'] and tcn_result['is_ai_generated']
    combined_confidence = (cnn_result['confidence'] + tcn_result['confidence']) / 2
    
    
    # Prepare result as a dictionary
    result = {
        "is_ai_generated": combined_is_ai,
        "confidence": combined_confidence,
        "cnn_result": cnn_result,
        "tcn_result": tcn_result,
        "features": features  # Include features if needed for further use
    }

    return result

def get_ai_generated_string(file_path):
    result = classify_audio(file_path)
    
    # Format the string output with each outcome on a new line
    output_string = (
        f"AI Generated: {result['is_ai_generated']}\n"
        f"Combined Confidence: {result['confidence']:.2f}\n"
        f"CNN Model Result: {result['cnn_result']}\n"
        f"TCN Model Result: {result['tcn_result']}"
    )
    
    return output_string

# New function that returns just "True" or "False"
def get_ai_generated_boolean_string(file_path):
    result = classify_audio(file_path)
    
    # Return "True" or "False" based on the AI generation result
    return "True" if result['is_ai_generated'] else "False"

if __name__ == "__main__":
    folder_path = "data"  # Replace with the actual path to the folder where audio will be saved
    file_name = "test_audio.wav"  # Replace with the expected file name
    audio_file = os.path.join(folder_path, file_name)
    
    # Wait for the audio file to appear (with a timeout mechanism if needed)
    while not os.path.exists(audio_file):
        print(f"Waiting for {audio_file} to be created...")
        time.sleep(2)  # Check every 2 seconds
    
    # Once the file is available, get the AI generated string result
    result_string = get_ai_generated_string(audio_file)
    print(result_string)  # Output the detailed results
    
    # Get just the "True" or "False" result
    boolean_result = get_ai_generated_boolean_string(audio_file)
    print(f"AI Generated (True/False): {boolean_result}")  # Output the boolean result
