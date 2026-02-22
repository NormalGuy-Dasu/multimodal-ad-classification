# 🎯 Multi-modal Online Advertisement Classification

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-F9AB00?logo=huggingface&logoColor=white)
![Flask](https://img.shields.io/badge/Web_App-Flask/Streamlit-000000?logo=flask&logoColor=white)

## 📌 Project Overview
This project implements a robust multi-modal AI system designed to classify online advertisements into **20 distinct categories**. By leveraging both visual and textual data (images, generated captions, extracted text via OCR, and metadata), the system achieves deep semantic understanding of complex ad content.

### ✨ Key Highlights & Performance
- **State-of-the-Art Models:** Integrates **CLIP**, **ViLT**, and **BLIP-2** for advanced feature extraction and multimodal fusion.
- **High Accuracy:** The fine-tuned CLIP model achieved an impressive **84.36% accuracy** on a rigorous test dataset of 5,929 samples.
- **Microservices Architecture:** The web demonstration is built using a decoupled architecture with independent services for Optical Character Recognition (OCR), Text Generation (Caption/Slogan), and the Main User Interface.

---

## 📂 Repository Structure

```text
source_code/
├── Source/
│   ├── Training/                 # Jupyter Notebooks for model fine-tuning
│   │   ├── clip_qa_cap_slo_ocr.ipynb
│   │   └── vilt_qa_cap_slo_ocr.ipynb
│   │
│   └── App_Demo/                 # Web Application & Microservices
│       ├── checkpoints/            # Pre-trained model weights (.pth)
│       ├── gen_service/            # BLIP-2 Generation Service
│       ├── ocr_service/            # PaddleOCR Service
│       └── main_app/               # Main UI and Inference integration
├── requirements_blip_env.txt
├── requirements_paddleocr_env.txt
└── requirements_torch_env.txt
```

---

## ⚙️ Environment Setup
To avoid dependency conflicts, this project requires three separate Conda virtual environments.

**1. OCR Service Environment (Requires Python 3.9)**

```bash
conda create -n env_ocr python=3.9
conda activate env_ocr
pip install -r requirements_paddleocr_env.txt
```

**2. Generation Service Environment (BLIP)**

```bash
conda create -n env_blip python=3.10
conda activate env_blip
pip install -r requirements_blip_env.txt
```

**3. Main App & PyTorch Environment (CLIP/ViLT)**

```bash
conda create -n env_main python=3.10
conda activate env_main
pip install -r requirements_torch_env.txt
```
---

## 🚀 Running the Web Demo
The application runs on three parallel services. Open three separate terminals (Anaconda Prompts), navigate to `source_code/Source/App_Demo`, and run the following commands simultaneously:

**Terminal 1: Start OCR Service**

```bash
conda activate env_ocr
cd ocr_service
python server.py
```

**Terminal 2: Start Generation Service**

```bash
conda activate env_blip
cd gen_service
python server.py
```

**Terminal 3: Start Main App UI**

```bash
conda activate env_main
cd main_app
python app.py
```

Once all services are running, open your web browser and navigate to the local URL provided in Terminal 3 (e.g., *`http://127.0.0.1:5000`*) to access the interactive demo.

---

## 👨‍💻 Authors
- [PhD. Luong Thi Ngoc Khanh] - Instructor
- [Tran Ba Dat] - Computer Science Student at Ton Duc Thang University
- [Nguyen Tan Phong] - Computer Science Student at Ton Duc Thang University

*This project was developed as part of the graduation thesis/coursework.*
