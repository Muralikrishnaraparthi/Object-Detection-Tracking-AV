Here is a comprehensive `README.md` file for your project, based on the code and report you provided.

---

# Real-Time Object Detection and Multi-Object Tracking for Autonomous Vehicles

**Author:** Muralikrishna Raparthi (2023AC05208)

**Date:** 2025-12-27

**Model Architecture:** YOLOv8n (Nano)

**Dataset:** KITTI Vision Benchmark Suite

## 📌 Project Overview

This project implements a real-time computer vision system designed for autonomous vehicles. It utilizes the **YOLOv8** architecture to detect and track objects such as cars, pedestrians, and cyclists in complex driving environments. The system provides a user-friendly web interface powered by **Gradio** for analyzing both static images and driving videos, complete with real-time analytics and performance metrics.

## 🚀 Key Features

* **Real-Time Object Detection**: High-speed inference using YOLOv8n on the KITTI dataset.


* **Multi-Object Tracking**: Capable of tracking objects across video frames with customizable Intersection over Union (IoU) and confidence thresholds.


* **Interactive Web Interface**: A Gradio-based dashboard to upload images/videos and view results instantly.


* **System Analytics**:
* **Object Counting**: Breakdown of detected classes (Car, Van, Truck, Pedestrian, etc.).


* **Performance Metrics**: Visualization of inference latency, confidence distribution, and tracker IoU.


* **Audit Report**: Automatic generation of system performance reports.




* **GPU Acceleration**: optimized for Tesla T4 hardware.



## 🛠️ Technology Stack

* **Language**: Python 3.10+
* **Deep Learning**: PyTorch, Ultralytics YOLOv8
* **Computer Vision**: OpenCV
* **Interface**: Gradio
* **Data Visualization**: Matplotlib, Seaborn

## 📂 Project Structure

```text
├── app.py                          # Main Gradio application entry point
├── requirements.txt                # Python dependencies
├── Dissertation_2023AC05208.ipynb  # Jupyter notebook for data ETL, EDA, and Model Training
├── Dissertation_Report.pdf         # Final system audit and performance report
└── best.pt                         # (Required) Trained YOLOv8 model weights

```

## 📦 Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd <repository-directory>

```


2. **Install dependencies:**
It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt

```



*Dependencies include: `ultralytics`, `gradio>=4.0.0`, `opencv-python-headless`, `torch`, `spaces`, `scipy`, `lapx`.* 


3. **Download/Place Weights:**
Ensure your trained model weights (`best.pt`) are in the root directory. If not found, the system will default to the standard `yolov8n.pt`.



## 🖥️ Usage

Run the application locally:

```bash
python app.py

```

Open your browser and navigate to the local URL provided (usually `http://127.0.0.1:7860`).

### Interface Guide

* **Image Analysis Tab**: Upload detection samples to see bounding boxes and class distributions.
* **Video Tracking Tab**: Upload driving footage to visualize tracking lines and object persistence.
* **System Info**: View hardware specifications and current model configuration.

## 📊 Model Performance

The system was audited on **2025-12-27** using a **Tesla T4** GPU.

* **Architecture**: YOLOv8n (Nano) - 3,012,408 parameters 


* **Input Resolution**: 640x640 px 


* **Training**: 50 Epochs, Batch Size 16, SGD Optimizer 


* **Validation Metrics**:
* **mAP @ 50% IoU**: 0.8394 


* **Precision**: 0.8407 

* **Inference Latency**: ~16.0ms (Mean) 





## 🏋️ Training Pipeline

The training process is documented in `Dissertation_2023AC05208.ipynb` and includes:

1. **Data Ingestion**: Automated fetching of the KITTI dataset from remote repositories.


2. **ETL Process**: Extraction and directory auditing to ensure data integrity.


3. **EDA**: Visualization of ground truth bounding boxes and class distributions.


4. **Training**: Fine-tuning YOLOv8 on the prepared dataset.

## 🤝 Acknowledgments

* **Ultralytics**: For the YOLOv8 framework.
* **KITTI Dataset**: For providing the vision benchmark suite.
* **Gradio**: For the web interface tools.
