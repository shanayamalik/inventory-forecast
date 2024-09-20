# Inventory Forecast

## Overview

This repository is designed for inventory management and replenishment. Utilizing advanced machine learning and statistical models, this system aids businesses in optimizing their inventory processes, ensuring efficient stock management and minimizing stockouts or overstock situations.

## Table of Contents
- [Overview](#overview)
- [Models](#models)
  - [Inventory Management](#inventory-management)
  - [Inventory Replenishment System](#inventory-replenishment-system)
- [Getting Started](#getting-started)
- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [License](#license)
- [Contributing](#contributing)

## Models

The repository incorporates various models to cater to different inventory scenarios and requirements:

### Inventory Management

- **Random Forest**: A versatile machine learning model known for its accuracy and ability to handle large datasets.
  - [Training Script](https://github.com/shanayamalik/inventory-forecast/blob/main/Inventory%20Management/Random%20Forest/training.py)
  - [Inference Script](https://github.com/shanayamalik/inventory-forecast/blob/main/Inventory%20Management/Random%20Forest/inference.py)

- **XGBoost**: An optimized gradient boosting machine learning library designed for speed and performance.
  - [Training Script](https://github.com/shanayamalik/inventory-forecast/blob/main/Inventory%20Management/XGBoost/training.py)
  - [Inference Script](https://github.com/shanayamalik/inventory-forecast/blob/main/Inventory%20Management/XGBoost/inference.py)

- **Statistical Model**: A model based on statistical methods tailored for time series forecasting.
  - [Training Script](https://github.com/shanayamalik/inventory-forecast/blob/main/Inventory%20Management/Statistical%20Model/training.py)
  - [Inference Script](https://github.com/shanayamalik/inventory-forecast/blob/main/Inventory%20Management/Statistical%20Model/inference.py)

### Inventory Replenishment System

- **Random Forest**:
  - [Training Script](https://github.com/shanayamalik/inventory-forecast/blob/main/Inventory%20Replenishment%20System/Random%20forest/training.py)
  - [Inference Script](https://github.com/shanayamalik/inventory-forecast/blob/main/Inventory%20Replenishment%20System/Random%20forest/inference.py)

- **Statistical Model**:
  - [Training Script](https://github.com/shanayamalik/inventory-forecast/blob/main/Inventory%20Replenishment%20System/Statistical%20Model/training.py)
  - [Inference Script](https://github.com/shanayamalik/inventory-forecast/blob/main/Inventory%20Replenishment%20System/Statistical%20Model/inference.py)

## Getting Started

To get started with the "inventory-forecast" system, follow the steps outlined in the setup section.

## Setup

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/shanayamalik/inventory-forecast.git
   cd inventory-forecast

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt

3. **Data Setup:**
     Ensure you have your data files placed in the appropriate directories. Update any paths in the training and inference scripts if necessary.

4. **Model Training:**
     Navigate to the desired model directory and run the training script
   ```bash
   python training.py

5. **Model Inference:**
     Once the model is trained, you can use the inference script to make predictions
   ```bash
   python inference.py

## License
This project is licensed under the MIT License. For more details, please refer to the LICENSE.md file.
