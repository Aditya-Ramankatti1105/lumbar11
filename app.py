import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Lumbar Disc Detection",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        r"D:\major project\lumbar11\lumbar11\best_lumbar_model.h5"
    )

model = load_model()

# ---------------- CLASS NAMES ----------------
CLASS_NAMES = ['bulging', 'degenerative', 'herniation', 'normal']

# ---------------- TITLE ----------------
st.title("🦴 Lumbar Spinal Cord Disc Detection System")
st.write("Upload a lumbar spine MRI image to predict the disc condition")

# ---------------- IMAGE UPLOADER ----------------
uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "png", "jpeg"]
)

# ---------------- PREDICTION ----------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # Preprocessing
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Display Result
    st.markdown("### 🧠 Prediction Result")
    st.success(f"**Predicted Condition:** {predicted_class.upper()}")
    st.info(f"**Confidence Score:** {confidence:.2f}%")

    # Probability Distribution
    st.markdown("### 📊 Class Probability Distribution")
    for cls, prob in zip(CLASS_NAMES, prediction[0]):
        st.write(f"{cls.capitalize()} : {prob * 100:.2f}%")
        st.progress(float(prob))

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption(
    "Note: This system performs real-time prediction on individual MRI images. "
    "Confusion matrix and detailed evaluation are performed during offline model training."
)
