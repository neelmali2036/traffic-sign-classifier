# 🚦 Traffic Sign Recognition System using Deep Learning

An end-to-end computer vision project that classifies 43 distinct road sign categories using a Convolutional Neural Network (CNN) trained on the German Traffic Sign Recognition Benchmark (GTSRB).

📌 Project Overview
Objective: Train an automated deep learning model to identify road signs in real-time under diverse environmental and lighting conditions.
Dataset: German Traffic Sign Recognition Benchmark (GTSRB)
Training Samples: 26,640 images
Test Samples: 12,630 images
Classes: 43 unique sign types
Model Architecture: Custom Multi-Layer CNN (Conv2D, ReLU, MaxPooling, Dropout, Dense)
Validation Accuracy: 97.02% on unseen test data
Deployment: Interactive GUI built with Gradio

🧠 Model Architecture
The network utilizes a sequential Convolutional Neural Network (CNN) designed for fine-grained image classification:
Block 1: 2 × Conv2D (32 filters, 3x3) → MaxPooling (2x2) → Dropout (0.25)
Block 2: 2 × Conv2D (64 filters, 3x3) → MaxPooling (2x2) → Dropout (0.25)
Classification Head: Flatten → Dense (256, ReLU) → Dropout (0.5) → Dense (43, Softmax)

📊 Results & Performance
Test Accuracy: 97.02%
Optimizer: Adam
Loss Function: Categorical Crossentropy
Epochs: 10

🛠️ Tech Stack
Deep Learning Framework: TensorFlow / Keras
Dataset Pipeline: Hugging Face Datasets
Image Processing: OpenCV, Pillow
Data Analysis & Visualization: NumPy, Matplotlib, Scikit-learn
Deployment Interface: Gradio

💻 How to Run This Project

1. Clone the Repository
Bash
git clone [https://github.com/neelmali2036/traffic-sign-classifier.git](https://github.com/neelmali2036/traffic-sign-classifier.git)
cd traffic-sign-classifier

3. Install Dependencies
Bash
pip install -r requirements.txt

5. Run Inference Web App
Bash
python app.py

