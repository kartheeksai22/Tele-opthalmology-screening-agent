# Tele-Ophthalmology Screening Assistant: Diabetic Retinopathy Screening System

## Overview
Tele-Ophthalmology Screening Assistant is an AI-powered screening tool designed to detect Diabetic Retinopathy from fundus images. Our Vision Transformer (ViT-B/16) model achieved 81.4% accuracy on 3,662 real clinical images (APTOS 2019 dataset). This repository contains the complete frontend, backend, and training scripts.

## Project Structure
- `app.py`: The Flask backend exposing the prediction API.
- `index.html`: The modern frontend UI for uploading images and viewing results.
- `train.py`: PyTorch training script to train the Baseline CNN, ResNet-50, or ViT-B/16 models on the APTOS 2019 dataset.
- `requirements.txt`: Python package dependencies.

## Prerequisites

To run and debug this model, you need to install standard Python data science and machine learning libraries.

### 1. Install VS Code Extensions
For the best development experience in VS Code, install the following extensions:
- **Python** (by Microsoft): Provides IntelliSense, linting, and formatting.
- **Python Debugger** (by Microsoft): Required for running the debugging server.
- **Jupyter** (by Microsoft): Useful if you decide to explore the data in `.ipynb` notebooks later.
- **Live Server** (by Ritwick Dey): Useful for serving `index.html` locally without CORS issues.

### 2. Install Python Dependencies
Create a virtual environment (optional but recommended) and install dependencies:

```bash
pip install flask flask-cors torch torchvision pillow transformers pandas timm tqdm
```

### 3. Running the Backend App

To start the AI backend server, run:
```bash
python app.py
```
This will start the Flask server at `http://127.0.0.1:5000`. The first time you run this, it will download the pre-trained ViT model weights (~300MB) from Hugging Face automatically.

### 4. Running the Frontend

With the backend running, just open `index.html` in your browser. 
Alternatively, and for best results, right-click `index.html` in VS Code and select **Open with Live Server** to serve the UI locally.

Upload a fundus image (retinal scan) and the API will return the severity and confidence score.

## Training the Models (Optional)

If you want to train the models yourself using the 3,662 images:

1. Download the APTOS 2019 Blindness Detection dataset from Kaggle to a local folder named `aptos2019/`.
   ```bash
   kaggle competitions download -c aptos2019-blindness-detection
   unzip aptos2019-blindness-detection.zip -d aptos2019
   ```

2. Run the `train.py` script specifying the model you wish to train:
   ```bash
   python train.py --model vit_b_16
   ```
   *Available models: `cnn`, `resnet50`, `vit_b_16`*

*(Note: Training the ViT model requires a CUDA-capable NVIDIA GPU. Without it, training will happen on the CPU and will be extremely slow. The backend prediction, however, runs perfectly fine on a CPU.)*
