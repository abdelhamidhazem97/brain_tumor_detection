"""
This script generates the brain_tumor_detection.ipynb notebook.
Run: python create_notebook.py
"""
import json, os

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": [source]}

def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source if isinstance(source, list) else [source]}

cells = []

# ---- Section 0 (Colab Setup) ----
cells.append(md("## 0. Colab Setup (Mount Google Drive & Extract Dataset)"))
cells.append(md("### طريقة التشغيل على Colab:\n1. قم بضغط المجلد `brain_tumor_dataset` الموجود في جهازك إلى ملف بصيغة `zip` (تأكد أن اسمه `brain_tumor_dataset.zip`).\n2. قم برفع الملف `brain_tumor_dataset.zip` إلى حسابك في **Google Drive** (في المجلد الرئيسي).\n3. قم بتشغيل الخلية التالية لربط حساب Google Drive وفك ضغط البيانات."))
cells.append(code([
    "## Mount Google Drive\n",
    "import os\n",
    "try:\n",
    "    from google.colab import drive\n",
    "    drive.mount('/content/drive')\n",
    "    IN_COLAB = True\n",
    "    print('Google Drive mounted successfully!')\n",
    "except:\n",
    "    IN_COLAB = False\n",
    "    print('Not running in Google Colab.')\n",
    "\n",
    "## Unzip the dataset from Google Drive\n",
    "if IN_COLAB:\n",
    "    zip_path = '/content/drive/MyDrive/brain_tumor_dataset.zip'\n",
    "    if os.path.exists(zip_path):\n",
    "        print('Extracting dataset...')\n",
    "        !unzip -q \"{zip_path}\" -d /content/\n",
    "        print('Dataset extracted to /content/brain_tumor_dataset')\n",
    "    else:\n",
    "        print(f'Error: Could not find {zip_path}. Please upload the zip file to your Google Drive.')\n"
]))

# ---- Section 1 ----
cells.append(md("## 1. Import the main libraries"))
cells.append(code([
    "## Major Libraries\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "import seaborn as sns\n",
    "\n",
    "## Deep Learning\n",
    "import tensorflow as tf\n",
    "from tensorflow import keras\n",
    "from tensorflow.keras import layers\n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "from tensorflow.keras.applications import VGG16\n",
    "from tensorflow.keras.models import Model, Sequential\n",
    "from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, BatchNormalization\n",
    "from tensorflow.keras.optimizers import Adam\n",
    "from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint\n",
    "\n",
    "## Other\n",
    "import os, cv2, warnings\n",
    "from sklearn.metrics import classification_report, confusion_matrix\n",
    "from collections import Counter\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "print('TensorFlow version:', tf.__version__)\n",
    "print('GPU Available:', tf.config.list_physical_devices('GPU'))"
]))

# ---- Section 2 ----
cells.append(md("## 2. Load data and look at the big picture"))
cells.append(code([
    "## Set paths\n",
    "if 'IN_COLAB' in globals() and IN_COLAB:\n",
    "    DATASET_DIR = '/content/brain_tumor_dataset'\n",
    "else:\n",
    "    DATASET_DIR = os.path.join(os.getcwd(), 'brain_tumor_dataset')\n",
    "\n",
    "TRAIN_DIR = os.path.join(DATASET_DIR, 'Training')\n",
    "TEST_DIR = os.path.join(DATASET_DIR, 'Testing')\n",
    "\n",
    "## Configuration\n",
    "IMG_SIZE = (224, 224)\n",
    "BATCH_SIZE = 32\n",
    "EPOCHS = 30\n",
    "LEARNING_RATE = 0.0001\n",
    "NUM_CLASSES = 4\n",
    "CLASS_NAMES = sorted(os.listdir(TRAIN_DIR))\n",
    "print('Classes:', CLASS_NAMES)"
]))
cells.append(code([
    "## Count images per class\n",
    "data_info = []\n",
    "for split in ['Training', 'Testing']:\n",
    "    split_dir = os.path.join(DATASET_DIR, split)\n",
    "    for cls in sorted(os.listdir(split_dir)):\n",
    "        cls_dir = os.path.join(split_dir, cls)\n",
    "        if os.path.isdir(cls_dir):\n",
    "            count = len(os.listdir(cls_dir))\n",
    "            data_info.append({'Split': split, 'Class': cls, 'Count': count})\n",
    "\n",
    "df_info = pd.DataFrame(data_info)\n",
    "df_info"
]))
cells.append(md("* > The dataset has **5600 training** and **1600 testing** images\n* > There are **4 classes**: glioma, meningioma, notumor, pituitary\n* > The dataset is **perfectly balanced** (equal samples per class)"))

# ---- Section 3 ----
cells.append(md("## 3. Exploratory Data Analysis (EDA)"))
cells.append(md("### Class Distribution"))
cells.append(code([
    "## Plot class distribution\n",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "for idx, split in enumerate(['Training', 'Testing']):\n",
    "    subset = df_info[df_info['Split'] == split]\n",
    "    colors = ['#FF6B6B', '#FFA502', '#2ED573', '#1E90FF']\n",
    "    bars = axes[idx].bar(subset['Class'], subset['Count'], color=colors, edgecolor='black', alpha=0.85)\n",
    "    axes[idx].set_title(f'{split} Set Distribution', fontsize=14, fontweight='bold')\n",
    "    axes[idx].set_xlabel('Class', fontsize=12)\n",
    "    axes[idx].set_ylabel('Count', fontsize=12)\n",
    "    for bar, val in zip(bars, subset['Count']):\n",
    "        axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,\n",
    "                       str(val), ha='center', fontweight='bold', fontsize=12)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
]))

cells.append(md("### Sample Images"))
cells.append(code([
    "## Display sample images from each class\n",
    "fig, axes = plt.subplots(4, 5, figsize=(16, 12))\n",
    "fig.suptitle('Sample Brain MRI Images by Class', fontsize=16, fontweight='bold')\n",
    "\n",
    "for row, cls in enumerate(CLASS_NAMES):\n",
    "    cls_dir = os.path.join(TRAIN_DIR, cls)\n",
    "    images = os.listdir(cls_dir)[:5]\n",
    "    for col, img_name in enumerate(images):\n",
    "        img_path = os.path.join(cls_dir, img_name)\n",
    "        img = cv2.imread(img_path)\n",
    "        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
    "        img = cv2.resize(img, IMG_SIZE)\n",
    "        axes[row][col].imshow(img)\n",
    "        axes[row][col].set_title(cls, fontsize=10)\n",
    "        axes[row][col].axis('off')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
]))

cells.append(md("### Image Size Analysis"))
cells.append(code([
    "## Analyze image dimensions\n",
    "widths, heights = [], []\n",
    "for cls in CLASS_NAMES:\n",
    "    cls_dir = os.path.join(TRAIN_DIR, cls)\n",
    "    for img_name in os.listdir(cls_dir)[:100]:  ## sample 100 per class\n",
    "        img = cv2.imread(os.path.join(cls_dir, img_name))\n",
    "        if img is not None:\n",
    "            h, w = img.shape[:2]\n",
    "            widths.append(w)\n",
    "            heights.append(h)\n",
    "\n",
    "print(f'Width  - Min: {min(widths)}, Max: {max(widths)}, Mean: {np.mean(widths):.0f}')\n",
    "print(f'Height - Min: {min(heights)}, Max: {max(heights)}, Mean: {np.mean(heights):.0f}')\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n",
    "axes[0].hist(widths, bins=30, color='steelblue', edgecolor='black')\n",
    "axes[0].set_title('Image Width Distribution', fontsize=13)\n",
    "axes[0].set_xlabel('Width (pixels)')\n",
    "axes[1].hist(heights, bins=30, color='coral', edgecolor='black')\n",
    "axes[1].set_title('Image Height Distribution', fontsize=13)\n",
    "axes[1].set_xlabel('Height (pixels)')\n",
    "plt.tight_layout()\n",
    "plt.show()"
]))

# ---- Section 4 ----
cells.append(md("## 4. Data Preprocessing & Augmentation"))
cells.append(code([
    "## Training data with augmentation\n",
    "train_datagen = ImageDataGenerator(\n",
    "    rescale=1.0/255.0,\n",
    "    rotation_range=20,\n",
    "    width_shift_range=0.2,\n",
    "    height_shift_range=0.2,\n",
    "    shear_range=0.2,\n",
    "    zoom_range=0.2,\n",
    "    horizontal_flip=True,\n",
    "    fill_mode='nearest',\n",
    "    validation_split=0.2\n",
    ")\n",
    "\n",
    "## Testing data - only rescaling\n",
    "test_datagen = ImageDataGenerator(rescale=1.0/255.0)\n",
    "\n",
    "## Create generators\n",
    "train_generator = train_datagen.flow_from_directory(\n",
    "    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,\n",
    "    class_mode='categorical', subset='training', shuffle=True, seed=42\n",
    ")\n",
    "\n",
    "val_generator = train_datagen.flow_from_directory(\n",
    "    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,\n",
    "    class_mode='categorical', subset='validation', shuffle=False, seed=42\n",
    ")\n",
    "\n",
    "test_generator = test_datagen.flow_from_directory(\n",
    "    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,\n",
    "    class_mode='categorical', shuffle=False\n",
    ")\n",
    "\n",
    "print(f'Training samples: {train_generator.samples}')\n",
    "print(f'Validation samples: {val_generator.samples}')\n",
    "print(f'Test samples: {test_generator.samples}')\n",
    "print(f'Class indices: {train_generator.class_indices}')"
]))

cells.append(md("### Visualize Augmented Images"))
cells.append(code([
    "## Show augmented samples\n",
    "images, labels = next(train_generator)\n",
    "class_labels = list(train_generator.class_indices.keys())\n",
    "\n",
    "fig, axes = plt.subplots(2, 4, figsize=(14, 7))\n",
    "fig.suptitle('Augmented Training Images', fontsize=14, fontweight='bold')\n",
    "for i, ax in enumerate(axes.flat):\n",
    "    if i < len(images):\n",
    "        ax.imshow(images[i])\n",
    "        ax.set_title(class_labels[np.argmax(labels[i])], fontsize=11)\n",
    "        ax.axis('off')\n",
    "plt.tight_layout()\n",
    "plt.show()"
]))

# ---- Section 5 ----
cells.append(md("## 5. Build the Model (Transfer Learning - VGG16)"))
cells.append(code([
    "## Load VGG16 pre-trained on ImageNet\n",
    "base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))\n",
    "\n",
    "## Freeze base layers\n",
    "for layer in base_model.layers:\n",
    "    layer.trainable = False\n",
    "\n",
    "## Unfreeze last 4 layers for fine-tuning\n",
    "for layer in base_model.layers[-4:]:\n",
    "    layer.trainable = True\n",
    "\n",
    "## Build classification head\n",
    "model = Sequential([\n",
    "    base_model,\n",
    "    GlobalAveragePooling2D(),\n",
    "    Dense(512, activation='relu'),\n",
    "    BatchNormalization(),\n",
    "    Dropout(0.5),\n",
    "    Dense(256, activation='relu'),\n",
    "    BatchNormalization(),\n",
    "    Dropout(0.3),\n",
    "    Dense(NUM_CLASSES, activation='softmax')\n",
    "])\n",
    "\n",
    "## Compile\n",
    "model.compile(\n",
    "    optimizer=Adam(learning_rate=LEARNING_RATE),\n",
    "    loss='categorical_crossentropy',\n",
    "    metrics=['accuracy']\n",
    ")\n",
    "\n",
    "model.summary()"
]))

# ---- Section 6 ----
cells.append(md("## 6. Train the Model"))
cells.append(code([
    "## Callbacks\n",
    "callbacks = [\n",
    "    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),\n",
    "    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),\n",
    "    ModelCheckpoint('models/brain_tumor_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)\n",
    "]\n",
    "\n",
    "## Train\n",
    "history = model.fit(\n",
    "    train_generator,\n",
    "    epochs=EPOCHS,\n",
    "    validation_data=val_generator,\n",
    "    callbacks=callbacks,\n",
    "    verbose=1\n",
    ")"
]))

# ---- Section 7 ----
cells.append(md("## 7. Training History Visualization"))
cells.append(code([
    "## Plot accuracy and loss curves\n",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "## Accuracy\n",
    "axes[0].plot(history.history['accuracy'], label='Training', linewidth=2)\n",
    "axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)\n",
    "axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')\n",
    "axes[0].set_xlabel('Epoch')\n",
    "axes[0].set_ylabel('Accuracy')\n",
    "axes[0].legend(fontsize=11)\n",
    "axes[0].grid(True, alpha=0.3)\n",
    "\n",
    "## Loss\n",
    "axes[1].plot(history.history['loss'], label='Training', linewidth=2)\n",
    "axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)\n",
    "axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')\n",
    "axes[1].set_xlabel('Epoch')\n",
    "axes[1].set_ylabel('Loss')\n",
    "axes[1].legend(fontsize=11)\n",
    "axes[1].grid(True, alpha=0.3)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
]))

# ---- Section 8 ----
cells.append(md("## 8. Evaluate the Model"))
cells.append(code([
    "## Evaluate on test set\n",
    "test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)\n",
    "print(f'\\nTest Loss: {test_loss:.4f}')\n",
    "print(f'Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)')"
]))

# ---- Section 9 ----
cells.append(md("## 9. Confusion Matrix & Classification Report"))
cells.append(code([
    "## Get predictions\n",
    "test_generator.reset()\n",
    "predictions = model.predict(test_generator, verbose=1)\n",
    "predicted_classes = np.argmax(predictions, axis=1)\n",
    "true_classes = test_generator.classes\n",
    "class_names = list(test_generator.class_indices.keys())\n",
    "\n",
    "## Classification Report\n",
    "print('Classification Report:')\n",
    "print('=' * 60)\n",
    "print(classification_report(true_classes, predicted_classes, target_names=class_names, digits=4))\n",
    "\n",
    "## Confusion Matrix\n",
    "cm = confusion_matrix(true_classes, predicted_classes)\n",
    "fig, ax = plt.subplots(figsize=(8, 6))\n",
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n",
    "            xticklabels=class_names, yticklabels=class_names, ax=ax, linewidths=0.5)\n",
    "ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')\n",
    "ax.set_xlabel('Predicted', fontsize=12)\n",
    "ax.set_ylabel('Actual', fontsize=12)\n",
    "plt.tight_layout()\n",
    "plt.show()"
]))

# ---- Section 10 ----
cells.append(md("## 10. Save the Model"))
cells.append(code([
    "## Save the trained model\n",
    "os.makedirs('models', exist_ok=True)\n",
    "model.save('models/brain_tumor_model.h5')\n",
    "print('Model saved locally to: models/brain_tumor_model.h5')\n",
    "\n",
    "## If in Colab, also copy the model to Google Drive\n",
    "if 'IN_COLAB' in globals() and IN_COLAB:\n",
    "    import shutil\n",
    "    drive_model_dir = '/content/drive/MyDrive/BrainTumorModels'\n",
    "    os.makedirs(drive_model_dir, exist_ok=True)\n",
    "    drive_model_path = os.path.join(drive_model_dir, 'brain_tumor_model.h5')\n",
    "    shutil.copy('models/brain_tumor_model.h5', drive_model_path)\n",
    "    print(f'\\n✅ Model safely copied to Google Drive at: {drive_model_path}')\n",
    "    print('👉 الآن يمكنك تحميل الموديل من جوجل درايف ووضعه في مجلد models لتشغيل الـ Web App')\n",
    "\n",
    "print(f'\\nFinal Test Accuracy: {test_accuracy*100:.2f}%')\n",
    "print('To run web interface locally: python app.py')"
]))

# ---- Section 11 ----
cells.append(md("## 11. Test with a Single Image"))
cells.append(code([
    "## Test prediction on a single image\n",
    "from tensorflow.keras.preprocessing.image import load_img, img_to_array\n",
    "\n",
    "def predict_single_image(img_path):\n",
    "    img = load_img(img_path, target_size=IMG_SIZE)\n",
    "    img_array = img_to_array(img) / 255.0\n",
    "    img_array = np.expand_dims(img_array, axis=0)\n",
    "    pred = model.predict(img_array, verbose=0)\n",
    "    pred_class = class_names[np.argmax(pred[0])]\n",
    "    confidence = np.max(pred[0]) * 100\n",
    "    return pred_class, confidence, pred[0]\n",
    "\n",
    "## Test on random images from test set\n",
    "fig, axes = plt.subplots(2, 4, figsize=(16, 8))\n",
    "fig.suptitle('Predictions on Test Images', fontsize=14, fontweight='bold')\n",
    "\n",
    "for i, ax in enumerate(axes.flat):\n",
    "    cls = class_names[i % len(class_names)]\n",
    "    cls_dir = os.path.join(TEST_DIR, cls)\n",
    "    img_name = os.listdir(cls_dir)[i]\n",
    "    img_path = os.path.join(cls_dir, img_name)\n",
    "    \n",
    "    pred_class, confidence, _ = predict_single_image(img_path)\n",
    "    img = cv2.imread(img_path)\n",
    "    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
    "    ax.imshow(img)\n",
    "    color = 'green' if pred_class == cls else 'red'\n",
    "    ax.set_title(f'True: {cls}\\nPred: {pred_class} ({confidence:.1f}%)', fontsize=9, color=color)\n",
    "    ax.axis('off')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
]))

# Build notebook
notebook = {
    "nbformat": 4,
    "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": cells
}

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'brain_tumor_detection.ipynb')
with open(path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)
print(f'Notebook created: {path}')
