import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from model.preprocess import preprocess_image
import cv2
from datetime import datetime
# from fpdf import FPDF
import io

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/cnn_model.h5")

model = load_model()

class_names = ['Melanoma', 'Clogged pores', 'Basal Cell Carcinoma', 'Inflammation', 'Dermatitis', 'pimples', 'Chronic dry Eczema', 'itchy Eczema', 'caly skin Eczema', 'Psoriasis']  

def generate_gradcam(model, image_array, class_index):
    import cv2
    from tensorflow.keras.models import Model

    # Automatically find the last Conv2D layer
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break
    else:
        raise ValueError("No Conv2D layer found in the model.")

    # Grad-CAM Model
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Forward and backward pass
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.expand_dims(image_array, axis=0))
        loss = predictions[:, class_index]

    # Compute gradients
    grads = tape.gradient(loss, conv_outputs)

    if grads is None or len(grads.shape) != 4:
        raise ValueError(f"Gradients shape invalid for Grad-CAM: {grads.shape}")

    grads = grads[0]  # remove batch dimension
    conv_outputs = conv_outputs[0]

    # Pooling the gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    # Normalize
    heatmap = np.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap + 1e-6)

    return heatmap.numpy()


# def generate_pdf(predicted_class, confidence, prob_df, date_str):
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Arial", "B", 16)
#     pdf.cell(0, 10, "AI Skin Disease Diagnosis Report", ln=True, align='C')
#     pdf.set_font("Arial", "", 12)
#     pdf.ln(10)

#     pdf.cell(0, 10, f"Date: {date_str}", ln=True)
#     pdf.cell(0, 10, f"Predicted Disease: {predicted_class}", ln=True)
#     pdf.cell(0, 10, f"Confidence: {confidence:.2f}%", ln=True)
#     pdf.ln(5)

#     pdf.cell(0, 10, "All Class Probabilities:", ln=True)
#     for _, row in prob_df.iterrows():
#         pdf.cell(0, 10, f"{row['Disease']}: {row['Confidence (%)']:.2f}%", ln=True)

#     pdf.ln(10)
#     pdf.multi_cell(0, 10, "This is not a clinical diagnosis. Please consult a certified dermatologist.")

#     # Export PDF to BytesIO
#     pdf_output = io.BytesIO()
#     pdf_bytes = pdf.output(dest='S').encode('latin1')  # Export as string then encode
#     pdf_output.write(pdf_bytes)
#     pdf_output.seek(0)
#     return pdf_output


def app():
    st.title("🧪 AI Skin Disease Diagnosis")
    st.markdown("Upload a dermoscopic image to get an **AI-powered diagnosis**. This tool uses a trained CNN model.")

    uploaded_file = st.file_uploader("📁 Upload a dermoscopic image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📸 Image Analyzed", width=300)

        if st.button("🔍 Diagnose"):
            try:
                # Preprocess and predict
                processed_img = preprocess_image(image)
                prediction = model.predict(np.expand_dims(processed_img, axis=0))[0]
                predicted_index = np.argmax(prediction)
                predicted_class = class_names[predicted_index]
                confidence = prediction[predicted_index] * 100

                # Display results
                st.subheader("📋 Diagnosis Result")
                st.markdown(f"🩺 **Predicted Disease**: `{predicted_class}`")
                st.markdown(f"📊 **Confidence**: `{confidence:.2f}%`")

                # Display bar chart
                st.subheader("🔎 Multi-Class Probability Breakdown")
                prob_df = pd.DataFrame({
                    "Disease": class_names,
                    "Confidence (%)": prediction * 100
                })
                st.bar_chart(prob_df.set_index("Disease"))

                # Grad-CAM
                try:
                    st.subheader("🧠 Model Attention Heatmap (Grad-CAM)")
                    heatmap = generate_gradcam(model, processed_img, predicted_index)
                    heatmap = cv2.resize(heatmap, image.size)
                    heatmap = np.uint8(255 * heatmap)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    superimposed_img = cv2.addWeighted(np.array(image), 0.6, heatmap, 0.4, 0)
                    st.image(superimposed_img, caption="Model Attention via Grad-CAM", use_container_width=True)
                except Exception as e:
                    st.warning(f"Grad-CAM not available: {e}")

                # Generate PDF
                # date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # pdf_file = generate_pdf(predicted_class, confidence, prob_df, date_str)

                # # Download button (only appears if PDF was successfully generated)
                # st.subheader("📄 Download Report")
                # st.download_button(
                #     "📥 Download Diagnosis Report (PDF)",
                #     data=pdf_file,
                #     file_name="diagnosis_report.pdf",
                #     mime="application/pdf"
                # )

                # Try another image
                if st.button("🔁 Try Another Image"):
                    st.experimental_rerun()

            except Exception as e:
                st.error(f"Something went wrong: {e}")

