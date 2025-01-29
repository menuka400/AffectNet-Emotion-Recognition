# AffectNet-Emotion-Recognition

This project implements a real-time emotion detection system using a deep learning model trained on the AffectNet dataset. The model is capable of recognizing eight emotions: **Angry, Disgust, Fear, Happiness, Neutral, Sadness, Surprise, and Contempt**.

## Features
- Real-time facial emotion recognition using OpenCV.
- Deep learning model trained on the AffectNet dataset.
- Supports multiple face detection in a single frame.
- Pretrained model (`.h5` format) for quick deployment.

## Dataset
This model is trained using the **AffectNet** dataset, a large-scale facial expression dataset containing annotated images of different human emotions. The dataset was preprocessed by resizing images to **96x96** pixels and normalizing pixel values.

## Installation
To set up the project, first install the required dependencies:

```bash
pip install -r requirements.txt
```

## Training the Model
To train the model from scratch, run the following script:

```bash
python train.py
```

This will:
1. Load and preprocess images from the AffectNet dataset.
2. Train a deep learning model with **Convolutional Neural Networks (CNNs)**.
3. Save the trained model as `emotiondetector.h5`.

## Running Real-Time Emotion Detection
To run real-time emotion detection using a webcam, execute:

```bash
python realtime_emotion_detection.py
```

## Model Architecture
The model consists of multiple convolutional layers, batch normalization, dropout, and fully connected layers. It uses the **Softmax** activation function in the final layer to classify emotions.

## Dependencies
The following libraries are required for the project:
- TensorFlow/Keras
- OpenCV
- NumPy
- Pandas
- Scikit-learn

## Author
This project was developed as part of an emotion recognition system using deep learning techniques.

---

# requirements.txt

```
tensorflow
keras
opencv-python
numpy
pandas
scikit-learn
matplotlib
tqdm
```

Ensure you have **Python 3.7+** installed before installing dependencies.

