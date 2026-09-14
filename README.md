# 🍎 Food Freshness Detection Using Deep Learning and Explainable AI

## 📌 Project Overview

Food Freshness Detection is a deep learning-based computer vision project that classifies fruit images as fresh or rotten.

The project uses **MobileNetV2** for image classification and **Grad-CAM** for Explainable AI.

A Streamlit web application provides an interactive interface where users can upload a fruit image and receive a prediction with confidence scores, probability distribution, and a visual Grad-CAM explanation.

---

## 🎯 Objectives

- Detect whether a fruit is fresh or rotten.
- Identify the type of fruit.
- Provide prediction confidence.
- Display probability scores for all classes.
- Provide an explainable AI visualization using Grad-CAM.
- Develop an easy-to-use web interface using Streamlit.

---

## 🧠 Model

The project uses **MobileNetV2**, a lightweight convolutional neural network suitable for image classification.

### Input Size

224 × 224 pixels

### Number of Classes

10

### Supported Fruits

- Apple
- Banana
- Mango
- Orange
- Strawberry

Each fruit has two freshness categories:

- Fresh
- Rotten

---

## 📊 Model Performance

The model achieved approximately **97.2% validation accuracy** during evaluation.

Performance on external images may vary depending on image quality, lighting, background, camera angle, and differences between the training dataset and real-world images.

---

## 💡 Explainable AI

The project uses **Grad-CAM (Gradient-weighted Class Activation Mapping)** to provide a visual explanation of the model's prediction.

The heatmap highlights image regions that contributed to the predicted class.

---

## 🔄 Project Workflow

1. Upload a fruit image.
2. Resize the image to 224 × 224 pixels.
3. Pass the image to the MobileNetV2 model.
4. Generate class probabilities.
5. Identify the predicted fruit and freshness status.
6. Display the confidence score.
7. Generate a Grad-CAM explanation.

---

## 🖥️ Application Screenshots

### Main Interface

![Main Interface](screenshots/main-interface.png)

### Prediction Result

![Prediction Result](screenshots/prediction.png)

### Grad-CAM Explanation

![Grad-CAM Explanation](screenshots/gradcam.png)

### Project Workflow

![Project Workflow](screenshots/workflow.png)

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- Streamlit
- NumPy
- Pillow
- Matplotlib
- Grad-CAM

---

## 📁 Project Structure

```text
food-freshness-detection/
│
├── app.py
├── app_backup.py
├── best_fruit_freshness_model.keras
├── Food_Freshness_Detection_Final (1).ipynb
├── gradcam_test.py
├── requirements.txt
├── README.md
├── .gitignore
└── screenshots/
    ├── main-interface.png
    ├── prediction.png
    ├── gradcam.png
    └── workflow.png