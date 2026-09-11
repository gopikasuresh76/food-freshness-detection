import os

import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Food Freshness Detection",
    page_icon="🍎",
    layout="wide"
)


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_FOLDER = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    PROJECT_FOLDER,
    "best_fruit_freshness_model.keras"
)


# =========================================================
# CLASS NAMES
# =========================================================

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


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        st.error(
            "Model file was not found.\n\n"
            f"Expected location:\n{MODEL_PATH}"
        )

        st.stop()

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model


model = load_model()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🍎 About the Project")

    st.write(
        "Food Freshness Detection is a deep learning "
        "application that classifies fruit images as "
        "fresh or rotten."
    )

    st.markdown("---")

    st.subheader("🧠 Model")

    st.write("MobileNetV2")

    st.subheader("📊 Classes")

    st.write("10 fruit freshness classes")

    st.subheader("💡 Explainability")

    st.write("Grad-CAM")

    st.markdown("---")

    st.subheader("🛠️ Technologies")

    st.write(
        "Python\n\n"
        "TensorFlow / Keras\n\n"
        "MobileNetV2\n\n"
        "Streamlit\n\n"
        "Grad-CAM"
    )

    st.markdown("---")

    st.caption(
        "Deep Learning Project"
    )


# =========================================================
# PROFESSIONAL HEADER
# =========================================================

st.markdown(
    """
<div style="text-align: center; padding: 20px 10px 10px 10px;">

<h1 style="font-size: 42px; margin-bottom: 5px;">
🍎 Food Freshness Detection
</h1>

<p style="font-size: 19px; margin-top: 0px;">
AI-powered fruit freshness classification using MobileNetV2
</p>

<p style="font-size: 15px;">
🔍 Computer Vision | 🧠 Deep Learning | 💡 Explainable AI
</p>

</div>
""",
    unsafe_allow_html=True
)


st.info(
    "📌 Upload a fruit image to predict whether it is "
    "fresh or rotten. The application also provides "
    "a Grad-CAM explanation showing the image regions "
    "that influenced the prediction."
)


# =========================================================
# MODEL INFORMATION
# =========================================================

st.subheader("⚙️ Model Information")

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:

    st.metric(
        "Model",
        "MobileNetV2"
    )

with info_col2:

    st.metric(
        "Input Size",
        "224 × 224"
    )

with info_col3:

    st.metric(
        "Classes",
        "10"
    )


# =========================================================
# IMAGE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📷 Upload a fruit image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    help=(
        "Upload a clear image of an apple, banana, "
        "mango, orange, or strawberry."
    )
)

st.caption(
    "Supported formats: JPG, JPEG, PNG"
)


# =========================================================
# PROCESS IMAGE
# =========================================================

if uploaded_file is not None:

    # -----------------------------------------------------
    # OPEN IMAGE
    # -----------------------------------------------------

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # =====================================================
    # DISPLAY UPLOADED IMAGE
    # =====================================================

    st.subheader("📷 Uploaded Image")

    st.image(
        image,
        use_container_width=True
    )


    # =====================================================
    # PREPROCESS IMAGE
    # =====================================================

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


    # =====================================================
    # NORMAL PREDICTION
    # =====================================================

    with st.spinner("Analyzing image..."):

        predictions = model.predict(
            img_array,
            verbose=0
        )


    predicted_index = int(
        np.argmax(
            predictions[0]
        )
    )


    predicted_class = class_names[
        predicted_index
    ]


    confidence = (
        float(
            predictions[0][predicted_index]
        ) * 100
    )


    # =====================================================
    # DETERMINE FRUIT AND FRESHNESS
    # =====================================================

    if predicted_class.startswith("Fresh"):

        freshness = "FRESH"
        status_icon = "✅"

    else:

        freshness = "ROTTEN"
        status_icon = "⚠️"


    fruit_name = predicted_class.replace(
        "Fresh",
        ""
    ).replace(
        "Rotten",
        ""
    )


    # =====================================================
    # PREDICTION RESULT
    # =====================================================

    st.subheader("🔍 Prediction Result")


    result_col1, result_col2 = st.columns(2)


    with result_col1:

        if freshness == "FRESH":

            st.success(
                f"### {status_icon} {freshness}"
            )

        else:

            st.warning(
                f"### {status_icon} {freshness}"
            )


    with result_col2:

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )


    # -----------------------------------------------------
    # DETECTED FRUIT
    # -----------------------------------------------------

    st.info(
        f"🍎 **Detected Fruit:** {fruit_name}"
    )


    # -----------------------------------------------------
    # FINAL RESULT MESSAGE
    # -----------------------------------------------------

    if freshness == "FRESH":

        st.success(
            f"✅ The **{fruit_name}** appears to be **FRESH**."
        )

    else:

        st.warning(
            f"⚠️ The **{fruit_name}** appears to be **ROTTEN**."
        )


    # =====================================================
    # PREDICTION PROBABILITIES
    # =====================================================

    st.subheader(
        "📊 Prediction Probabilities"
    )


    probability_data = {}


    for i in range(
        len(class_names)
    ):

        probability_data[
            class_names[i]
        ] = float(
            predictions[0][i] * 100
        )


    sorted_probabilities = sorted(
        probability_data.items(),
        key=lambda x: x[1],
        reverse=True
    )


    # -----------------------------------------------------
    # DISPLAY PROBABILITY BARS
    # -----------------------------------------------------

    for class_name, probability in sorted_probabilities:

        col1, col2 = st.columns(
            [2, 5]
        )

        with col1:

            st.write(
                f"**{class_name}**"
            )

        with col2:

            st.progress(
                min(
                    max(
                        float(probability) / 100,
                        0.0
                    ),
                    1.0
                ),
                text=f"{probability:.2f}%"
            )


    # =====================================================
    # GRAD-CAM
    # =====================================================

    st.subheader(
        "🧠 Explainable AI - Grad-CAM"
    )


    st.write(
        "Grad-CAM highlights the image regions "
        "that contributed to the model's prediction."
    )


    try:

        # -------------------------------------------------
        # FIND MOBILENETV2
        # -------------------------------------------------

        base_model = model.get_layer(
            "mobilenetv2_1.00_224"
        )


        # -------------------------------------------------
        # FIND LAST CONVOLUTIONAL LAYER
        # -------------------------------------------------

        last_conv_layer = base_model.get_layer(
            "Conv_1"
        )


        # -------------------------------------------------
        # CREATE GRAD-CAM MODEL
        # -------------------------------------------------

        grad_model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=[
                last_conv_layer.output,
                base_model.output
            ]
        )


        # -------------------------------------------------
        # CALCULATE GRADIENTS
        # -------------------------------------------------

        with tf.GradientTape() as tape:

            conv_outputs, base_output = grad_model(
                img_array,
                training=False
            )


            # ---------------------------------------------
            # RECREATE CLASSIFIER
            # ---------------------------------------------

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


            grad_predictions = x


            # ---------------------------------------------
            # USE SAME PREDICTED CLASS
            # ---------------------------------------------

            class_channel = grad_predictions[
                :,
                predicted_index
            ]


        # -------------------------------------------------
        # CALCULATE GRADIENTS
        # -------------------------------------------------

        grads = tape.gradient(
            class_channel,
            conv_outputs
        )


        # -------------------------------------------------
        # GLOBAL AVERAGE POOLING
        # -------------------------------------------------

        pooled_grads = tf.reduce_mean(
            grads,
            axis=(0, 1, 2)
        )


        # -------------------------------------------------
        # CREATE HEATMAP
        # -------------------------------------------------

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


        # =================================================
        # DISPLAY GRAD-CAM
        # =================================================

        st.write(
            f"**Explanation for:** {predicted_class}"
        )


        # -------------------------------------------------
        # SIDE-BY-SIDE DISPLAY
        # -------------------------------------------------

        original_col, gradcam_col = st.columns(2)


        # -------------------------------------------------
        # ORIGINAL IMAGE
        # -------------------------------------------------

        with original_col:

            st.markdown(
                "### 📷 Original Image"
            )

            st.image(
                image,
                use_container_width=True
            )


        # -------------------------------------------------
        # GRAD-CAM IMAGE
        # -------------------------------------------------

        with gradcam_col:

            st.markdown(
                "### 🔥 Grad-CAM Explanation"
            )


            fig = plt.figure(
                figsize=(6, 5)
            )


            plt.imshow(
                image
            )


            plt.imshow(
                heatmap,
                cmap="jet",
                alpha=0.45,
                extent=(
                    0,
                    image.width,
                    image.height,
                    0
                )
            )


            plt.title(
                f"Grad-CAM - {predicted_class}"
            )


            plt.axis(
                "off"
            )


            plt.tight_layout()


            st.pyplot(
                fig
            )


            plt.close(
                fig
            )


        st.success(
            "✅ Grad-CAM explanation generated successfully."
        )


    except Exception as e:

        st.warning(
            "Prediction worked, but Grad-CAM "
            "could not be generated."
        )

        st.code(
            str(e)
        )

# =========================================================
# HOW IT WORKS
# =========================================================

st.markdown("---")

st.subheader("🔄 How It Works")

step1, step2, step3, step4 = st.columns(4)

with step1:
    st.markdown("### 1️⃣ Upload")
    st.write(
        "Upload a clear image of a fruit "
        "in JPG, JPEG, or PNG format."
    )

with step2:
    st.markdown("### 2️⃣ Analyze")
    st.write(
        "The MobileNetV2 deep learning model "
        "analyzes the uploaded image."
    )

with step3:
    st.markdown("### 3️⃣ Predict")
    st.write(
        "The system predicts the fruit type "
        "and whether it is fresh or rotten."
    )

with step4:
    st.markdown("### 4️⃣ Explain")
    st.write(
        "Grad-CAM highlights the image regions "
        "that influenced the prediction."
    )
# =========================================================
# SUPPORTED FRUITS
# =========================================================

st.markdown("---")

st.subheader("🍎 Supported Fruits")

fruit_col1, fruit_col2, fruit_col3, fruit_col4, fruit_col5 = st.columns(5)

with fruit_col1:
    st.markdown("### 🍎")
    st.write("**Apple**")

with fruit_col2:
    st.markdown("### 🍌")
    st.write("**Banana**")

with fruit_col3:
    st.markdown("### 🥭")
    st.write("**Mango**")

with fruit_col4:
    st.markdown("### 🍊")
    st.write("**Orange**")

with fruit_col5:
    st.markdown("### 🍓")
    st.write("**Strawberry**")

st.caption(
    "Each fruit has separate fresh and rotten classes."
)
# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.markdown("---")

st.subheader(
    "📈 Model Performance"
)


performance_col1, performance_col2, performance_col3 = st.columns(3)


with performance_col1:

    st.metric(
        "Validation Accuracy",
        "97.2%"
    )


with performance_col2:

    st.metric(
        "Model",
        "MobileNetV2"
    )


with performance_col3:

    st.metric(
        "Output Classes",
        "10"
    )


st.caption(
    "Validation accuracy obtained during model evaluation. "
    "Performance on external images may vary depending "
    "on image quality and dataset differences."
)

# =========================================================
# LIMITATIONS
# =========================================================

st.markdown("---")

st.subheader("⚠️ Limitations")

st.write(
    "The model was trained on a specific fruit image dataset. "
    "Prediction performance may vary for images captured in "
    "different lighting conditions, backgrounds, camera angles, "
    "or image qualities."
)

st.write(
    "The prediction should be considered an AI-based estimation "
    "and not a replacement for human inspection."
)
# =========================================================
# FOOTER
# =========================================================

st.markdown("---")


st.caption(
    "Food Freshness Detection using "
    "MobileNetV2 and Explainable AI (Grad-CAM)"
)