from meta_ai_api import MetaAI
from feature_extraction import extract_features_for_llm, format_prompt_for_llm


def get_llm_response(audio_file):
    
    ai = MetaAI()

    # Extract features from the audio file
    features_summary = extract_features_for_llm(audio_file)

    # Format the prompt for the LLM
    prompt = format_prompt_for_llm(features_summary)

    # Use MetaAI to prompt for a response
    response = ai.prompt(message=prompt) 

    # Extract only the 'message' field from the response
    message_only = response.get('message', '')

    return message_only

if __name__ == "__main__":
    # Path to the audio file
    audio_file = "data\\test_audio.wav" 
    