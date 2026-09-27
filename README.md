# ISRO-BAH-
# Satellite Image Super-Resolution and Colorization

## Overview

This project develops an AI-based pipeline for enhancing satellite imagery by combining **Super-Resolution (SR)** and **Image Colorization** techniques.

The objective is to reconstruct high-resolution, visually interpretable satellite images from lower-resolution and limited-band satellite observations. The proposed approach uses separate deep-learning models for spatial resolution enhancement and spectral/color reconstruction.

## Problem Statement

Satellite missions often provide imagery with different spatial and spectral resolutions. High-resolution multispectral imagery can be computationally expensive or unavailable for certain observations.

This project addresses the problem of:

1. Enhancing the spatial resolution of low-resolution satellite imagery.
2. Reconstructing visually meaningful RGB imagery from limited spectral-band information.
3. Producing high-quality satellite images while preserving important spatial and spectral features.

## Proposed Approach

The complete pipeline consists of two independent deep-learning models:

```text
Low-Resolution Satellite Image
            │
            ▼
   Super-Resolution Model
            │
            ▼
 High-Resolution Image
            │
            ▼
   Colorization Model
            │
            ▼
 Reconstructed RGB Image
1. Super-Resolution Model
The Super-Resolution model takes low-resolution satellite imagery as input and learns to reconstruct a higher-resolution representation.
The model is trained using paired low-resolution and high-resolution satellite images.
The objective is to minimize the difference between the reconstructed image and the corresponding high-resolution reference image.
Evaluation metrics include:
- Mean Squared Error (MSE)
- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index (SSIM)
2. Colorization Model
The colorization stage reconstructs RGB information from satellite data with limited spectral information.
A deep-learning architecture based on U-Net / ResNet-style feature extraction is used to learn the mapping between input satellite bands and the corresponding RGB representation.
The model uses encoder-decoder feature extraction to preserve both:
- Global contextual information
- Fine spatial details
The colorization model is evaluated using image reconstruction metrics such as:
- MSE
- PSNR
- SSIM
Dataset
The project uses satellite imagery associated with the ISRO Bharatiya Antariksh Hackathon (BAH) problem statement.
The dataset contains satellite observations with different spatial/spectral characteristics that are used to train and evaluate the proposed models.
The dataset is not included in this repository due to size and/or distribution constraints.
Model Architecture
Super-Resolution
The Super-Resolution component uses a dedicated deep-learning architecture optimized for reconstruction of high-resolution satellite imagery.
Input Low-Resolution Image
            │
            ▼
     Feature Extraction
            │
            ▼
   Deep Feature Learning
            │
            ▼
   Upsampling / Reconstruction
            │
            ▼
    High-Resolution Output

Colorization
The colorization component uses an encoder-decoder architecture with residual feature extraction.
Satellite Input
      │
      ▼
    Encoder
      │
      ├── Feature Extraction
      │
      ▼
Residual / Deep Features
      │
      ▼
    Decoder
      │
      ▼
 Reconstructed RGB Image

Technologies Used
- Python
- PyTorch
- NumPy
- OpenCV
- Pillow
- Matplotlib
- Scikit-learn
- CUDA / NVIDIA GPU acceleration
- Google Colab / Local GPU environment
Project Structure
Satellite-Image-Enhancement/
│
├── data/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── models/
│   ├── super_resolution/
│   └── colorization/
│
├── notebooks/
│
├── scripts/
│   ├── train_sr.py
│   ├── train_colorization.py
│   └── inference.py
│
├── results/
│   ├── super_resolution/
│   └── colorization/
│
├── weights/
│   └── trained_model_weights
│
├── requirements.txt
└── README.md

Note: Dataset files and trained model weights may be excluded from the repository because of their size.

Installation
Clone the repository:
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <REPOSITORY_NAME>

Create a virtual environment:
python -m venv venv

Activate the environment on Windows:
venv\Scripts\activate

Install the required dependencies:
pip install -r requirements.txt

Training
Train the Super-Resolution Model
python train_sr.py

Train the Colorization Model
python train_colorization.py

Modify the dataset paths and training parameters according to the local setup.
Inference
After downloading or training the required model weights, inference can be performed using:
python inference.py

The script generates enhanced satellite imagery using the trained models.
Evaluation
The models are evaluated using quantitative image-quality metrics.
MSE
Measures the average squared difference between the predicted and reference images.
Lower MSE indicates lower reconstruction error.
PSNR
Measures the signal-to-noise ratio between the reconstructed and reference images.
Higher PSNR generally indicates better reconstruction quality.
SSIM
Measures structural similarity between the reconstructed and reference images.
Values closer to 1 indicate greater structural similarity.
Results
The developed models demonstrate the ability to:
- Enhance spatial details in low-resolution satellite imagery.
- Reconstruct visually meaningful RGB representations.
- Preserve important spatial structures during image enhancement.
- Generate outputs suitable for visual interpretation and further downstream analysis.
Example outputs can be found in:
results/

Applications
The proposed pipeline can support applications such as:
- Satellite image enhancement
- Earth observation
- Remote sensing
- Environmental monitoring
- Land-cover analysis
- Agricultural monitoring
- Disaster assessment
- Geospatial analysis
- Satellite-based visual interpretation
Future Work
Future improvements could include:
- Incorporating perceptual loss functions.
- Exploring GAN- and diffusion-based super-resolution methods.
- Incorporating multispectral and hyperspectral information.
- Improving spectral fidelity during colorization.
- Testing on additional satellite datasets.
- Developing lightweight models for onboard or near-real-time processing.
- Quantitative evaluation using additional remote-sensing-specific metrics.
Authors
Parth Tripathi
M.Tech Chemical Engineering
Indian Institute of Technology Hyderabad
Acknowledgements
This project was developed as part of the ISRO Bharatiya Antariksh Hackathon and focuses on applying deep learning techniques to satellite image enhancement and reconstruction.
