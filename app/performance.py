# evaluate.py

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    roc_curve,
    auc
)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Constants
DATA_DIR = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MODEL_PATH = "model/cnn_model.h5"
HISTORY_PATH = "model/history.png"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    else:
        return None

@st.cache_resource
def get_val_generator():
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    return datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

def plot_multiclass_roc(y_true, y_prob, class_labels):
    n_classes = len(class_labels)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], lw=2,
                label=f'{class_labels[i]} (AUC = {roc_auc[i]:.2f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Multiclass ROC Curve')
    ax.legend(loc="lower right")
    st.pyplot(fig)

def calculate_sensitivity_specificity(cm):
    """Returns DataFrame of sensitivity and specificity per class"""
    sensitivity = []
    specificity = []
    for i in range(len(cm)):
        TP = cm[i, i]
        FN = sum(cm[i, :]) - TP
        FP = sum(cm[:, i]) - TP
        TN = cm.sum() - (TP + FP + FN)
        sens = TP / (TP + FN) if TP + FN > 0 else 0
        spec = TN / (TN + FP) if TN + FP > 0 else 0
        sensitivity.append(sens)
        specificity.append(spec)
    return pd.DataFrame({
        'Class': list(range(len(cm))),
        'Sensitivity (Recall)': np.round(sensitivity, 3),
        'Specificity': np.round(specificity, 3)
    })

def app():
    st.title("📈 Model Evaluation")
    st.markdown("Evaluate the performance of the trained CNN model on validation data.")

    st.markdown("### 📖 What Do These Metrics Mean?")
    st.markdown("""
    - **Accuracy**: The proportion of correctly classified samples.
    - **Sensitivity (Recall)**: Ability of the model to detect actual positives.
    - **Specificity**: Ability of the model to detect actual negatives.
    - **AUC-ROC**: Shows how well the model can distinguish between classes.
    - **Confusion Matrix**: Table showing true vs predicted classes.
    - **Precision / F1-score**: Quality of positive predictions.
    """)

    model = load_model()
    if model is None:
        st.error("❌ Trained model not found. Please train and save the model first.")
        return

    val_generator = get_val_generator()
    class_labels = list(val_generator.class_indices.keys())

    if not class_labels:
        st.warning("⚠️ No classes found in validation data. Please check your dataset structure.")
        return

    st.info("🔄 Generating predictions...")
    y_true = val_generator.classes
    y_prob = model.predict(val_generator)
    y_pred = np.argmax(y_prob, axis=1)

    # 📋 Classification Report
    st.subheader("📋 Classification Report")
    report_dict = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df.round(3))

    # ✅ Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    st.success(f"✅ Validation Accuracy: {accuracy:.4f}")

    # 🧪 AUC-ROC Score & Curve
    st.subheader("📊 AUC-ROC Curve")
    try:
        y_true_onehot = to_categorical(y_true, num_classes=len(class_labels))
        auc_score = roc_auc_score(y_true_onehot, y_prob, multi_class='ovr')
        st.success(f"AUC-ROC (macro-average): {auc_score:.4f}")
        plot_multiclass_roc(y_true_onehot, y_prob, class_labels)
    except Exception as e:
        st.warning(f"⚠️ AUC-ROC computation error: {str(e)}")

    # 🔍 Confusion Matrix
    st.subheader("🔍 Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig_cm, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig_cm)

    # 📊 Sensitivity & Specificity Table
    st.subheader("📌 Sensitivity and Specificity per Class")
    sens_spec_df = calculate_sensitivity_specificity(cm)
    sens_spec_df["Class"] = [class_labels[i] for i in sens_spec_df["Class"]]
    st.dataframe(sens_spec_df.set_index("Class"))

    # 📉 Accuracy/Loss Graph
    st.subheader("📉 Training Accuracy & Loss Over Epochs")
    if os.path.exists(HISTORY_PATH):
        st.image(HISTORY_PATH, caption="Training Performance")
    else:
        st.info("Training graph not found. Please save the training history plot as 'model/history.png'.")
