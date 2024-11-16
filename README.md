Multimodal AI Audio and Image Detector
This project is a multimodal detection system designed to identify AI-generated audio and images using advanced machine learning models. The system integrates audio and image classification models within a PyQt5 application, providing an interactive interface for users to upload and analyze media files.

Project Overview
The Multimodal AI Audio and Image Detector aims to:

Detect AI-generated audio using Temporal Convolutional Networks (TCN) and Convolutional Neural Networks (CNN).
Classify AI-generated images with a multi-classifier approach using ResNet-50, Vision Transformer (ViT), and VGG16 models.
Leverage the Meta 3.1 LLM API to enhance data processing, feature extraction, and classification accuracy across different media types.
Features
Audio Detection: Utilizes TCN and CNN models to analyze audio files and classify them as AI-generated or human-generated.
Image Detection: Employs ResNet-50, ViT, and VGG16 models for multi-class classification of images.
User-Friendly Interface: Built with PyQt5, allowing users to upload files and receive detection results through an intuitive GUI.
Technologies Used
Frameworks & Libraries: PyQt5, TensorFlow, PyTorch
Models: TCN, CNN, ResNet-50, ViT, VGG16
APIs: Meta 3.1 LLM API for advanced feature extraction
