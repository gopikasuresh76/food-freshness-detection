# Food Freshness Detection Using Deep Learning and Explainable AI

## Project Description

This project detects whether fruits are fresh or rotten using a deep learning model.

The application uses MobileNetV2 for image classification and Grad-CAM for Explainable AI.

## Classes

The model detects 10 classes:

- FreshApple
- FreshBanana
- FreshMango
- FreshOrange
- FreshStrawberry
- RottenApple
- RottenBanana
- RottenMango
- RottenOrange
- RottenStrawberry

## Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- Streamlit
- NumPy
- Pillow
- Matplotlib
- Grad-CAM

## Application Workflow

1. User uploads a fruit image.
2. The image is resized to 224 × 224 pixels.
3. MobileNetV2 processes the image.
4. The model predicts the fruit freshness class.
5. The confidence score is displayed.
6. Prediction probabilities are displayed.
7. Grad-CAM generates an explanation heatmap.

## How to Run

Activate the virtual environment:

    .\venv312\Scripts\Activate.ps1

Run the application:

    streamlit run app.py

Open the application in a browser using the URL shown in the terminal.

## Model

The trained model is stored as:

    best_fruit_freshness_model.keras