# 🍎 Food Freshness Detection Using Deep Learning and Explainable AI

A deep learning-based computer vision application that detects fruit freshness and identifies the fruit type using **MobileNetV2** and **Grad-CAM Explainable AI**.

The project provides an interactive **Streamlit web application** where users can upload a fruit image and receive a freshness prediction, confidence score, class probabilities, and visual explanation of the model's decision.

---

## 📌 Project Overview

Food Freshness Detection is a deep learning-based computer vision project designed to classify fruit images into different freshness categories.

The system uses **MobileNetV2**, a lightweight convolutional neural network, for image classification. **Grad-CAM (Gradient-weighted Class Activation Mapping)** is used to provide a visual explanation of the prediction.

The application supports five fruit types, with each fruit classified as either **Fresh** or **Rotten**.

---

## 🎯 Objectives

- Detect whether a fruit is fresh or rotten.
- Identify the type of fruit.
- Provide prediction confidence.
- Display probability scores for all supported classes.
- Provide an Explainable AI visualization using Grad-CAM.
- Develop an easy-to-use web interface using Streamlit.

---

## 🧠 Model

### MobileNetV2

The project uses **MobileNetV2** as the deep learning model for image classification.

MobileNetV2 is a lightweight convolutional neural network designed to provide efficient image classification while requiring relatively fewer computational resources.

### Model Configuration

| Parameter | Value |
|---|---|
| Model | MobileNetV2 |
| Input Size | 224 × 224 pixels |
| Number of Classes | 10 |
| Classification Type | Fresh / Rotten |
| Explainability | Grad-CAM |

### Supported Fruits

The model supports the following fruits:

- 🍎 Apple
- 🍌 Banana
- 🥭 Mango
- 🍊 Orange
- 🍓 Strawberry

Each fruit has two freshness categories:

- Fresh
- Rotten

---

## 📊 Model Performance

The model achieved approximately **97.2% validation accuracy** during evaluation.

This result represents performance on the validation data used during model evaluation.

Performance on external images may vary depending on:

- Lighting conditions
- Background
- Camera angle
- Image quality
- Image source
- Differences between training and real-world images

Therefore, the model should be considered a deep learning-based estimation rather than a replacement for human inspection.

---

## 💡 Explainable AI

The project uses **Grad-CAM (Gradient-weighted Class Activation Mapping)** to provide a visual explanation of the model's prediction.

Grad-CAM generates a heatmap showing the image regions that contributed most strongly to the predicted class.

This makes the model's decision more interpretable instead of providing only a prediction label.

### Example

The Streamlit application displays:

1. The original uploaded image.
2. The predicted fruit and freshness category.
3. The Grad-CAM heatmap showing important image regions.

---

## 🔄 Project Workflow

```text
User Uploads Fruit Image
          ↓
Image Preprocessing
          ↓
Resize Image to 224 × 224
          ↓
MobileNetV2 Model
          ↓
Class Prediction
          ↓
Fresh / Rotten + Fruit Type
          ↓
Confidence & Probability Scores
          ↓
Grad-CAM Explanation
          ↓
Display Results in Streamlit
```

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

- **Python**
- **TensorFlow**
- **Keras**
- **MobileNetV2**
- **Streamlit**
- **NumPy**
- **Pillow**
- **Matplotlib**
- **Grad-CAM**
- **Jupyter Notebook**

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
```

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/gopikasuresh76/food-freshness-detection.git
cd food-freshness-detection
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment on Windows

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install the required packages

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your web browser.

---

## ⚠️ Limitations

- The model was trained using a specific fruit image dataset.
- Prediction performance may vary on images captured in different environments.
- Differences in lighting, background, camera angle, and image quality can affect predictions.
- External images may produce different results from images belonging to the original dataset.
- The system provides an AI-based estimation and should not replace human inspection.

---

## 🚀 Future Improvements

- Support for additional fruits and vegetables.
- Use larger and more diverse datasets.
- Improve real-world image generalization.
- Deploy the application to a cloud platform.
- Develop a mobile application.
- Explore additional Explainable AI techniques.
- Improve robustness under different lighting and background conditions.

---

## 📌 Project Type

**Academic Deep Learning / Computer Vision Project**

### Key Concepts Demonstrated

- Image Classification
- Convolutional Neural Networks
- Transfer Learning
- MobileNetV2
- Explainable AI
- Grad-CAM
- Model Evaluation
- Streamlit Application Development

---

## 👩‍💻 Project Highlights

This project demonstrates the integration of **deep learning, computer vision, Explainable AI, and web application development** into a single practical application.

The combination of **MobileNetV2 for classification** and **Grad-CAM for visual explanation** helps make the prediction system both efficient and more interpretable.