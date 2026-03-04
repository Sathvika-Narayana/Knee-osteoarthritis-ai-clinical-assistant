🧠 Knee Osteoarthritis Severity Detection using Domain-Specific Transfer Learning and Explainable AI

📌 Overview

This project presents an AI-powered clinical assistant designed to analyze Knee X-ray images and predict the severity of Knee Osteoarthritis (KOA) using deep learning.

The system performs automated Kellgren–Lawrence (KL) grading and provides visual explanations using GradCAM, along with AI-generated clinical interpretation using a large language model.

The application is deployed as an interactive web-based medical AI tool accessible from both desktop and mobile browsers.

---

🏥 Problem Statement

Knee Osteoarthritis is one of the most common musculoskeletal disorders worldwide. Radiologists typically assess the severity using the Kellgren–Lawrence (KL) grading system, which requires expert interpretation of radiographic features.

Manual grading is:

- Time consuming
- Subjective
- Dependent on clinical expertise

This project aims to develop an AI-based assistive system capable of automatically grading KOA severity from knee X-ray images.

---

🧠 Domain-Specific Transfer Learning

The model leverages Domain-Specific Transfer Learning, a technique where pretrained convolutional neural networks are adapted to specialized tasks such as medical image analysis.

Instead of training from scratch, pretrained architectures such as:

- ResNet50
- DenseNet121

are fine-tuned to classify Knee Osteoarthritis severity levels.

Benefits of domain-specific transfer learning include:

- Faster convergence
- Improved performance with limited medical datasets
- Better feature extraction for radiographic patterns

The pretrained backbone learns general visual features while the classification head is customized for KL grade prediction.

---

🔍 Explainable AI with GradCAM

Medical AI systems require interpretability. To ensure transparency, this system integrates GradCAM (Gradient-weighted Class Activation Mapping).

GradCAM highlights the regions of the X-ray image that most influenced the model’s prediction, allowing clinicians to verify whether the model focuses on relevant anatomical structures.

This improves:

- Trust in AI predictions
- Model transparency
- Clinical usability

---

📊 KL Grading System

The system predicts the Kellgren–Lawrence severity grades:

KL Grade| Description
KL0| Normal knee joint
KL1| Doubtful osteoarthritis
KL2| Mild osteoarthritis
KL3| Moderate osteoarthritis
KL4| Severe osteoarthritis

---

🧩 System Architecture

The project follows a modular AI system architecture:

User Uploads X-ray Image
        ↓
Streamlit Web Interface
        ↓
FastAPI Backend
        ↓
Deep Learning Model (ResNet / DenseNet)
        ↓
GradCAM Visualization
        ↓
AI Clinical Explanation (Gemini API)

---

🛠 Technology Stack

Programming Language

- Python

Deep Learning Framework

- PyTorch
- Torchvision

Web Frameworks

- FastAPI (Backend API)
- Streamlit (Frontend UI)

Libraries

- OpenCV
- NumPy
- Matplotlib
- Pillow

AI Services

- Gemini API (Clinical explanation generation)

---

📂 Project Structure

koa_website
│
├── backend
│   ├── main.py
│   ├── model.py
│   ├── gradcam.py
│   ├── explanation.py
│   └── models
│       └── trained_model.pt
│
├── frontend
│   └── app.py
│
├── requirements.txt
└── README.md

---

🚀 Running the Project Locally

Clone Repository

git clone https://github.com/yourusername/koa-ai-assistant.git
cd koa-ai-assistant

Install Dependencies

pip install -r requirements.txt

Start Backend Server

cd backend
uvicorn main:app --reload

Start Frontend Application

cd frontend
streamlit run app.py

The application will launch in your browser.

---

📱 Mobile Compatibility

The system is designed as a web-based AI application, meaning it can be accessed from:

- Desktop browsers
- Mobile browsers
- Tablets

This allows clinicians or users to upload X-ray images directly from mobile devices.

---

⚠ Disclaimer

This project is intended for educational and research purposes only.
It should not be used as a replacement for professional medical diagnosis.

👩‍💻 Authors

1.Narayana Sathvika
2.B.V.S.Tejaswini
3.U.Avinash
