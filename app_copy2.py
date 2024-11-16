import os
import time
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from image_test import analyze_single_image, get_ai_generated_boolean_string_image
from test import  get_ai_generated_string, get_ai_generated_boolean_string
from LLM_audio import get_llm_response
from LLM_Image import check_if_image_is_ai_generated, setup_google_genai

class MyApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("test_1.ui", self)  # Load your .ui file

        # Ensure the 'data' folder exists
        os.makedirs("data", exist_ok=True)

        # Connect the push button to the upload action
        
        self.pushButton.clicked.connect(self.upload_file)
        

    def upload_file(self):
        self.textEdit_1.setText("Loading  ...")

        # Check which radio button is selected
        if self.radioButton_audio.isChecked():
            # Audio radio button selected, open dialog for MP3 or WAV
            file_filter = "Audio Files (*.mp3 *.wav)"
            save_name = "test_audio.wav"  # Name for the saved audio file
        elif self.radioButton_image.isChecked():
            # Image radio button selected, open dialog for PNG
            file_filter = "Image Files (*.png)"
            save_name = "test_image.png"  # Name for the saved image file
        else:
            # No radio button selected, show an error
            QMessageBox.warning(self, "Error", "Please select either Audio or Image option!")
            return

        # Open the file dialog for the selected type
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", file_filter, options=options)

        if file_path:
            # Save path where the file will be stored in the 'data' folder
            save_path = os.path.join("data", save_name)

            self.textEdit_1.setText("Upload Successful\n File has been saved in Folder")
            try:
                # Save the file to the data folder with the predefined name
                with open(file_path, "rb") as file_in:
                    with open(save_path, "wb") as file_out:
                        file_out.write(file_in.read())

                # If it's an audio file, process it with classify_audio
                if self.radioButton_audio.isChecked():
                    result = get_ai_generated_string(save_path)
                    result4 = get_llm_response(save_path)  # Pass the saved file path to classify_audio
                    self.textEdit_details.setText("LLM response: " + result4 + "\n" + result)
                    result2 = get_ai_generated_boolean_string(file_path)
                    self.textEdit_output.setText(f"AI DETECTED: {result2}")  # Display AI-generated string
                else:
                    # Initialize the Google Generative AI model using the API key
                    #api_key = "AIzaSyB6m5_xWp9esRMOA0wq08gaqPNx2vfrE48"
                    #model = setup_google_genai(api_key)

                    # For images, call the function that checks if the image is AI-generated
                    result = analyze_single_image(save_path)
                    self.textEdit_details.setText(f"AI Image Detection Result: {result}")  # Display AI-generated response
                    result4 = get_ai_generated_boolean_string_image(file_path)
                    self.textEdit_output.setText(f"AI DETECTED: {result4}")
                    
                return save_path  # Return the path where the file was saved
            except Exception as e:
                # Display any error during the file saving process
                QMessageBox.warning(self, "Error", f"File upload failed: {str(e)}")
        else:
            # No file selected, show an error
            QMessageBox.warning(self, "Error", "No file selected!")

        return None  # Return None if upload was unsuccessful
 # R    eturn None if upload was unsuccessful
    # R    eturn None if upload was unsuccessful

def run_gui():
    app = QtWidgets.QApplication([])
    window = MyApp()
    window.show()
    return app.exec_()

if __name__ == "__main__":
    run_gui()
