import google.generativeai as genai
import PIL.Image

# Function to set up the model
def setup_google_genai(api_key):
    genai.configure(api_key=api_key)
    
    generation_config = {
        "temperature": 0.9,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }

    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
    ]

    model = genai.GenerativeModel(model_name="gemini-1.5-flash",
                                  generation_config=generation_config,
                                  safety_settings=safety_settings)
    return model

# Function to get the response from Google GenerativeAI model
def check_if_image_is_ai_generated(image_path, model):
    try:
        # Load the image
        image = PIL.Image.open(image_path)
        
        # Prepare the prompt
        prompt = ("Is this image AI generated? Just say True or False.", image)
        
        # Get the response from the model
        response = model.generate_content(prompt)
        
        # Return the text response
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# Example usage
if __name__ == "__main__":
    api_key = #your api key
    image_path = "data\\test_image.jpg"

    # Setup the Google GenerativeAI model
    model = setup_google_genai(api_key)

    # Get and print the response
    result = check_if_image_is_ai_generated(image_path, model)
    print(result)
