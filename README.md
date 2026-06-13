# Strawberry Disease Classification using CNN & ResNet50

An image classification project that detects whether a strawberry is **fresh**, has **scorch** disease, or has **leaf spot** disease using deep learning models built with TensorFlow/Keras.

## Dataset

The dataset is organized into three class folders:

| Folder | Class | Description |
|---|---|---|
| `Strawberry_fresh` | Fresh | Healthy strawberry leaf images |
| `Strawberry_scrotch` | Scorch | Strawberry leaves affected by scorch disease |
| `Strawberry_spot` | Spot | Strawberry leaves affected by angular leaf spot |

The dataset is stored as a zip archive (`strawberry.zip`) on Google Drive and extracted at runtime.

## Models

The notebook implements two approaches:

### 1. Custom CNN (Baseline)

A simple convolutional neural network built from scratch.

- **Input size:** 180×180 pixels
- **Architecture:** 3 Conv2D + MaxPooling blocks → Flatten → Dense(128) → Dense(3)
- **Optimizer:** Adam
- **Loss:** Sparse Categorical Crossentropy
- **Epochs:** 10
- **Output:** `strawberry_classifier.h5`

### 2. ResNet50 Transfer Learning (Improved)

A fine-tuned ResNet50 model pretrained on ImageNet.

- **Input size:** 224×224 pixels
- **Architecture:** ResNet50 (frozen base) → GlobalAveragePooling2D → Dense(128) → Dropout(0.5) → Dense(3)
- **Data augmentation:** Random horizontal flip, rotation (±10%), zoom (±10%)
- **Training strategy:**
  - Phase 1 — Feature extraction with frozen ResNet50 base (10 epochs)
  - Phase 2 — Fine-tuning the last 30 ResNet layers with a lower learning rate of 1e-5 (10 epochs)
- **Early stopping:** Monitors validation loss with patience of 3 epochs
- **Output:** `strawberry_resnet50_final.h5`

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib

Install dependencies:

```bash
pip install tensorflow matplotlib numpy
```

## Usage

### Training

1. Upload `strawberry.zip` to your Google Drive. 
2. Open `main_model.ipynb` in Google Colab.
3. Run all cells — the notebook mounts Drive, extracts the dataset, trains both models, and saves them as `.h5` files.


### Dataset Google Drive Link: **[Download Dataset from Google Drive](https://drive.google.com/drive/folders/1fzFdVZvIJ3MsfLNzPxJfq7TzZccwNv-6?usp=sharing)**


### Inference

```python
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("strawberry_resnet50_final.h5")
class_names = ["Strawberry_fresh", "Strawberry_scrotch", "Strawberry_spot"]

img = image.load_img("path/to/image.jpg", target_size=(224, 224))
img_array = preprocess_input(tf.expand_dims(image.img_to_array(img), 0))

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print(f"Predicted: {class_names[np.argmax(score)]} ({100 * np.max(score):.2f}%)")
```

## Project Structure

```
├── main_model.ipynb              # Training notebook (Colab)
├── strawberry.zip                # Dataset archive (on Google Drive)
│   ├── Strawberry_fresh/         # Fresh leaf images
│   ├── Strawberry_scrotch/       # Scorch-diseased leaf images
│   └── Strawberry_spot/          # Spot-diseased leaf images
├── strawberry_classifier.h5      # Saved baseline CNN model
├── strawberry_resnet50_final.h5  # Saved fine-tuned ResNet50 model
└── README.md
```

## License

This project is for educational and research purposes.
