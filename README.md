# Inventory Forecast

The "inventory-forecast" repository is dedicated to providing solutions for inventory management and replenishment using various machine learning and statistical models. The repository is structured into two main sections: "Inventory Management" and "Inventory Replenishment System". Each section contains implementations using Random Forest, XGBoost, and Statistical Models.

## Inventory Management

### Random Forest
- **Training**: The training script for the Random Forest model can be found [here](https://github.com/shanayamalik/inventory-forecast/blob/main/Inventory%20Management/Random%20Forest/training.py). This script is responsible for training the model using historical inventory data.
- **Inference**: The inference script can be found [here](https://github.com/shanayamalik/inventory-forecast/blob/main/Inventory%20Management/Random%20Forest/inference.py). It uses the trained Random Forest model to predict future inventory levels.

### XGBoost
- **Training**: The training script for the XGBoost model is located [here](https://github.com/shanayamalik/inventory-forecast/blob/main/Inventory%20Management/XGBoost/training.py).
- **Inference**: The inference script for the XGBoost model is available [here](https://github.com/shanayamalik/inventory-forecast/blob/main/Inventory%20Management/XGBoost/inference.py).

### Statistical Model
- **Training**: The training script for the Statistical model can be accessed [here](https://github.com/shanayamalik/inventory-forecast/blob/main/Inventory%20Management/Statistical%20Model/training.py).
- **Inference**: The inference script for the Statistical model is located [here](https://github.com/shanayamalik/inventory-forecast/blob/main/Inventory%20Management/Statistical%20Model/inference.py).

## Inventory Replenishment System

### Random Forest
- **Training**: The training script for the Random Forest model in the replenishment system is [here](https://github.com/shanayamalik/inventory-forecast/blob/main/Inventory%20Replenishment%20System/Random%20forest/training.py).
- **Inference**: The inference script for this model can be found [here](https://github.com/shanayamalik/inventory-forecast/blob/main/Inventory%20Replenishment%20System/Random%20forest/inference.py).

### Statistical Model
- **Training**: The training script for the Statistical model in the replenishment system is available [here](https://github.com/shanayamalik/inventory-forecast/blob/main/Inventory%20Replenishment%20System/Statistical%20Model/training.py).
- **Inference**: The inference script for this model is located [here](https://github.com/shanayamalik/inventory-forecast/blob/main/Inventory%20Replenishment%20System/Statistical%20Model/inference.py).

## Getting Started

To get started with the models, follow these steps:

1. Clone the repository to your local machine.
2. Navigate to the desired model directory.
3. Run the training script to train the model using your data.
4. Use the inference script to make predictions.

## Usage
Both projects are designed for easy command-line execution. 
For the Inventory Replenishment System:
```bash
python inference_script_name.py --date YYYY-MM-DD
```

For the Inventory Replenishment System:
```bash
python inference_script_name.py --start MM-DD-YYYY --duration D
```
Replace `inference_script_name.py` with the appropriate script name and provide the relevant date parameters.

---

## Contributing

Contributions are welcome! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute.

## License

This project is licensed under the MIT License. See the [LICENSE.md](LICENSE.md) file for details.
