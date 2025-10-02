import os
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.utils import to_categorical
import seaborn as sns
import numpy as np

# === CONFIG ===
DATA_DIR = "dataset"
MODEL_DIR = "model"
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "cnn_model.h5")
HISTORY_PATH = os.path.join(MODEL_DIR, "history.json")
PLOT_PATH = os.path.join(MODEL_DIR, "history.png")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# === Create output dir ===
os.makedirs(MODEL_DIR, exist_ok=True)

# === Data generators ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# === Base model ===
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

# === Custom classifier ===
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# === Compile model ===
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === Callbacks ===
callbacks = [
    EarlyStopping(patience=4, restore_best_weights=True),
    ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)
]

# === Train model ===
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

# === Save model ===
model.save(MODEL_SAVE_PATH)
print(f"✅ Model saved to: {MODEL_SAVE_PATH}")

# === Save history ===
with open(HISTORY_PATH, 'w') as f:
    json.dump(history.history, f)
print(f"📄 History saved to: {HISTORY_PATH}")

# === Plot training history ===
def plot_history(history_dict, save_path=None):
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Accuracy')
    plt.plot(epochs_range, val_acc, label='Val Accuracy')
    plt.legend()
    plt.title('Accuracy Over Epochs')

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Val Loss')
    plt.legend()
    plt.title('Loss Over Epochs')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"📊 Training plot saved to: {save_path}")
    plt.show()

plot_history(history.history, save_path=PLOT_PATH)

# === Model Evaluation ===
print("📈 Evaluating model...")

# Predictions
y_true = val_generator.classes
y_pred_prob = model.predict(val_generator)
y_pred = np.argmax(y_pred_prob, axis=1)

# Classification report
print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=list(val_generator.class_indices.keys())))

# AUC-ROC
# try:
#     auc = roc_auc_score(y_true, y_pred_prob, multi_class='ovr')
#     print(f"🎯 AUC-ROC (macro): {auc:.4f}")
# except Exception as e:
#     print(f"⚠️ AUC-ROC computation failed: {e}")
# Convert y_true to one-hot
y_true_onehot = to_categorical(y_true, num_classes=len(val_generator.class_indices))

try:
    auc = roc_auc_score(y_true_onehot, y_pred_prob, multi_class='ovr')
    print(f"AUC-ROC (macro): {auc:.4f}")
except Exception as e:
    print(f"⚠️ AUC-ROC computation failed: {e}")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d',
            xticklabels=val_generator.class_indices.keys(),
            yticklabels=val_generator.class_indices.keys())
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
