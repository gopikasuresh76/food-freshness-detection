import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ==========================================
# 1. PROJECT FOLDER
# ==========================================

PROJECT_FOLDER = os.path.dirname(
    os.path.abspath(__file__)
)


# ==========================================
# 2. FILE PATHS
# ==========================================

MODEL_PATH = os.path.join(
    PROJECT_FOLDER,
    "best_fruit_freshness_model.keras"
)

IMAGE_PATH = os.path.join(
    PROJECT_FOLDER,
    "fresh_apple.jpg"
)


# ==========================================
# 3. CLASS NAMES
# ==========================================

class_names = [
    "FreshApple",
    "FreshBanana",
    "FreshMango",
    "FreshOrange",
    "FreshStrawberry",
    "RottenApple",
    "RottenBanana",
    "RottenMango",
    "RottenOrange",
    "RottenStrawberry"
]


# ==========================================
# 4. LOAD MODEL
# ==========================================

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully!")


# ==========================================
# 5. LOAD IMAGE
# ==========================================

print("Loading image...")

image = Image.open(
    IMAGE_PATH
).convert("RGB")

original_image = np.array(image)

image_resized = image.resize(
    (224, 224)
)

img_array = np.array(
    image_resized
).astype("float32")

img_array = np.expand_dims(
    img_array,
    axis=0
)

print(
    "Image prepared:",
    img_array.shape
)


# ==========================================
# 6. NORMAL MODEL PREDICTION
# ==========================================

normal_predictions = model.predict(
    img_array,
    verbose=0
)

normal_index = np.argmax(
    normal_predictions[0]
)

normal_class = class_names[
    normal_index
]

normal_confidence = (
    float(
        normal_predictions[0][normal_index]
    ) * 100
)

print()
print("==============================")
print("NORMAL MODEL PREDICTION")
print("==============================")

print(
    "Predicted class:",
    normal_class
)

print(
    "Confidence:",
    f"{normal_confidence:.2f}%"
)

print("==============================")


# ==========================================
# 7. FIND MOBILENETV2
# ==========================================

base_model = model.get_layer(
    "mobilenetv2_1.00_224"
)

print(
    "Base model:",
    base_model.name
)


# ==========================================
# 8. FIND LAST CONVOLUTIONAL LAYER
# ==========================================

last_conv_layer = base_model.get_layer(
    "Conv_1"
)

print(
    "Last convolutional layer:",
    last_conv_layer.name
)


# ==========================================
# 9. BUILD GRAD-CAM MODEL
# ==========================================

grad_model = tf.keras.models.Model(
    inputs=base_model.input,
    outputs=[
        last_conv_layer.output,
        base_model.output
    ]
)

print(
    "Grad-CAM model created successfully!"
)


# ==========================================
# 10. CALCULATE GRADIENTS
# ==========================================

with tf.GradientTape() as tape:

    conv_outputs, base_output = grad_model(
        img_array,
        training=False
    )

    # Recreate the layers after MobileNetV2

    x = base_output

    start = False

    for layer in model.layers:

        if layer.name == "mobilenetv2_1.00_224":
            start = True
            continue

        if start:
            x = layer(
                x,
                training=False
            )

    predictions = x

    # IMPORTANT:
    # Use the prediction from the
    # normal model prediction

    class_channel = predictions[
        :,
        normal_index
    ]


# ==========================================
# 11. CALCULATE GRADIENTS
# ==========================================

grads = tape.gradient(
    class_channel,
    conv_outputs
)

print(
    "Gradients calculated successfully!"
)


# ==========================================
# 12. GLOBAL AVERAGE POOLING
# ==========================================

pooled_grads = tf.reduce_mean(
    grads,
    axis=(0, 1, 2)
)


# ==========================================
# 13. CREATE HEATMAP
# ==========================================

conv_outputs = conv_outputs[0]

heatmap = conv_outputs @ (
    pooled_grads[..., tf.newaxis]
)

heatmap = tf.squeeze(
    heatmap
)

heatmap = tf.maximum(
    heatmap,
    0
)

max_value = tf.reduce_max(
    heatmap
)

heatmap = heatmap / (
    max_value +
    tf.keras.backend.epsilon()
)

heatmap = heatmap.numpy()


# ==========================================
# 14. DISPLAY ORIGINAL IMAGE
# ==========================================

plt.figure(
    figsize=(7, 7)
)

plt.imshow(
    original_image
)

plt.title(
    f"Original Image\n"
    f"{normal_class} - "
    f"{normal_confidence:.2f}%"
)

plt.axis("off")

plt.tight_layout()

plt.show()


# ==========================================
# 15. DISPLAY GRAD-CAM
# ==========================================

plt.figure(
    figsize=(7, 7)
)

plt.imshow(
    heatmap,
    cmap="jet"
)

plt.title(
    f"Grad-CAM Explanation\n"
    f"Focus: {normal_class}"
)

plt.axis("off")

plt.tight_layout()

plt.show()


# ==========================================
# 16. FINISHED
# ==========================================

print()
print("==============================")
print("GRAD-CAM COMPLETED")
print("==============================")

print(
    "Model prediction:",
    normal_class
)

print(
    "Model confidence:",
    f"{normal_confidence:.2f}%"
)

print(
    "The heatmap shows the image regions "
    "that influenced the prediction."
)

print("==============================")