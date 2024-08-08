# CARLA-semantic-image-segmentation

![image](https://github.com/user-attachments/assets/ae67268c-9b06-4ee5-9333-13277ee46469)

## Table of Contents
1. [Introduction](#introduction)
2. [Installation Instructions](#installation-instructions)
    - [Prerequisites](#prerequisites)
    - [Dependencies](#dependencies)
    - [Setup](#setup)
3. [Usage](#usage)
    - [Running the Notebook](#running-the-notebook)
    - [Example Usage](#example-usage)
4. [Features](#features)
5. [Configuration](#configuration)

## Introduction

This repository contains a project focused on building a U-Net, a type of Convolutional Neural Network (CNN) designed for quick and precise image segmentation. The U-Net architecture is particularly effective for tasks that require precise localization, such as semantic image segmentation, where each pixel in an image is labeled with its corresponding class.

Semantic image segmentation is crucial for applications like self-driving cars, which require a pixel-perfect understanding of their environment to navigate safely. This project uses a dataset from the CARLA self-driving car simulator to implement and evaluate a U-Net model for segmenting images of urban driving scenarios.

This project achieves 4 main goals:
- Build and understand a U-Net architecture
- Explain the differences between regular CNNs and U-Nets
- Implement semantic image segmentation on the CARLA dataset
- Apply sparse categorical crossentropy for pixelwise predictions

## Installation Instructions

### Prerequisites
Before you begin, ensure you have the following installed:
- Python 3.7 or higher
- Jupyter Notebook or JupyterLab

### Dependencies
This project requires several Python libraries. The key dependencies include:
- TensorFlow
- NumPy
- Matplotlib
- OpenCV
- Scikit-learn
- Pillow
- h5py
- tqdm
- glob2

### Setup
Clone the repository to your local machine:
```bash
git clone https://github.com/justinliu23/CARLA-semantic-image-segmentation.git
cd CARLA-semantic-image-segmentation
```

Open `Image_segmentation_Unet_v2.ipynb` in Jupyter Notebook to start exploring the project.

## Usage

### Running the Notebook
To use this project, open the Jupyter Notebook `Image_segmentation_Unet_v2.ipynb` in your Jupyter environment. The notebook contains step-by-step instructions and code cells to execute the image segmentation task.

### Example Usage
1. **Load and Prepare the Dataset:**
    - The notebook includes cells to download and preprocess the CARLA dataset. Make sure to execute these cells to prepare the data for training.

2. **Build the U-Net Model:**
    - Follow the notebook instructions to define the U-Net architecture using TensorFlow/Keras.

3. **Train the Model:**
    - Execute the training cells to train your U-Net model on the preprocessed CARLA dataset.

4. **Evaluate the Model:**
    - Use the provided evaluation cells to assess the model's performance on the validation set.

5. **Predict and Visualize:**
    - Run the prediction cells to generate segmentation masks for test images and visualize the results.

## Features
- **U-Net Architecture:** An implementation of the U-Net model for semantic image segmentation.
- **Data Preprocessing:** Scripts to preprocess the CARLA self-driving car dataset.
- **Training and Evaluation:** Cells to train the U-Net model and evaluate its performance.
- **Visualization:** Tools to visualize the segmentation masks predicted by the model.

## Configuration
The project is designed to be flexible and configurable. Key configuration options include:
- **Model Parameters:** Adjust the number of filters, kernel size, and other hyperparameters in the model definition cells.
- **Training Parameters:** Configure batch size, number of epochs, learning rate, and other training parameters in the training cells.
- **Data Paths:** Set the paths to the dataset and any other required files in the data loading cells.
