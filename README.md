# Paper Waste Detection - Final Project

This repository contains the source code for my final object detection project at Satakunta University of Applied Sciences (SAMK). The goal of this project is to accurately detect and label various types of paper waste to assist in automated recycling processes.

## 🚀 Project Highlights
* **Model:** Faster R-CNN with ResNet-50 FPN backbone.
* **Hardware:** Trained an **NVIDIA RTX 3090 (24GB)** on the SAMK AI Server.
* **Best Validation Loss:** **0.0743** (Optimized with Weight Decay 5e-4 and Learning Rate 5e-5).

## 🛠️ Performance Optimization
Moving from a baseline on Google Colab (Nvidia T4) to a high-performance local server allowed for deeper optimization:
* Increased **Num Workers** to 8 for faster data loading.
* Extended training to **50 epochs**.
* Refined hyperparameters to prevent overfitting and achieve high precision (scores up to **0.99**).

## 📂 Project Structure
* `main.py`: Entry point for training.
* `model.py`: Architecture definition.
* `test_script.py`: Inference script for evaluating new images.
* `dataset.py`: Custom dataset and transformation logic.
* `sessions/`: Contains the training logs and session history.

## 🎓 Acknowledgments
Special thanks to my instructor, **Mitra Daneshmand**, for the guidance throughout this course and the AI community for the architectural support.
