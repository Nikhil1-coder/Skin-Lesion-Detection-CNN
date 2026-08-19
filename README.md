# 🩺 DermaAI - Skin Lesion Detection using CNN

An AI-powered web application for classifying skin lesion images as **Benign** or **Malignant** using a Convolutional Neural Network (CNN).

The application provides an interactive interface built with **Streamlit** and uses a trained TensorFlow/Keras deep learning model for image classification.

---

## 📌 Project Overview

Skin lesion classification is an important application of computer vision in medical image analysis.

This project uses a **Convolutional Neural Network (CNN)** to analyze dermoscopic skin-lesion images and classify them into two classes:

- 🟢 **Benign**
- 🔴 **Malignant**

The trained model is integrated into a Streamlit application where users can upload an image and receive a prediction with a confidence score.

> ⚠️ This project is intended for educational and demonstration purposes only. It is not a medical diagnostic tool.

---

## ✨ Features

- 🖼️ Upload skin-lesion images
- 🤖 CNN-based image classification
- 🟢 Benign / 🔴 Malignant prediction
- 📊 Prediction confidence score
- 📈 Class probability visualization
- 🕘 Recent prediction history
- 🗑️ Clear prediction history
- 📱 Interactive Streamlit interface
- ⚠️ Medical disclaimer

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow | Deep learning framework |
| Keras | CNN model development |
| Streamlit | Web application |
| NumPy | Numerical operations |
| Pillow | Image processing |

---

## 🧠 Model

The project uses a custom **Convolutional Neural Network (CNN)** for binary image classification.

### Model Input

```text
224 × 224 × 3