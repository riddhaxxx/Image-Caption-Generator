---
title: Image Caption Generator
emoji: 🖼️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
---
# Image-Caption-Generator

This project generates image captions using a hybrid deep learning architecture combining:

- YOLOv8 for object detection
- ResNet50 for image feature extraction
- LSTM for natural language caption generation
- Beam Search with Semantic Injection

## Features

- Upload any image
- Detect objects using YOLOv8
- Extract visual features using ResNet50
- Generate descriptive captions
- Display detected objects

## Project Structure

```
Image-Caption-Generator/
│── app.py
│── caption_model.keras
│── tokenizer.pkl
│── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

> **Note:** The YOLOv8 model weights (`yolov8m.pt`) are not included in this repository. Ultralytics will automatically download them over the internet the first time you run the app. Please ensure you have an active internet connection, otherwise the app will display an error.

## Model

- YOLOv8m
- ResNet50
- LSTM Caption Generator

## Developed Using

- Python
- TensorFlow
- Streamlit
- Ultralytics
