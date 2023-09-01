# Inventory Management Systems

This repository contains two main projects:
- **Inventory Management**
- **Inventory Replenishment System**

---

## 1. Inventory Management

### Overview
The Inventory Management system is designed to predict inventory levels up to a specified date. Multiple models were experimented with for this project, but the most effective model has been the Random Forest.

### Key Components

- **Training Script**: Trains the model using historical inventory data.
- **Inference Script**: Uses the trained model to predict inventory levels up to a provided date. This script accepts a command-line input date until which the inventory is to be predicted.

### Dataset
The dataset for this project can be found in the `inventory_management/dataset` folder.

### Best Model
- **Random Forest**

---

## 2. Inventory Replenishment System

### Overview
The Inventory Replenishment System is designed to forecast sales and recommend replenishment amounts to ensure stock availability for a specified duration.

### 3. Key Components

- **Training Script**: This script trains the forecasting model on historical sales data. It employs the Random Forest Regressor from the Scikit-Learn library, ensuring a balance of performance and accuracy.
- **Inference Script**: Here, we leverage the previously trained forecasting model to predict sales for a range of dates. These predictions then inform our calculations for the recommended restock amount, considering both the last available inventory and the total forecasted sales.

### 4. Dataset
The dataset for this project can be found in the `inventory_replenishment_system/dataset` folder.

### 5. Best Model
- **Random Forest**

---

## 6. Usage
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

## 6. License
This project is licensed under the terms of the XYZ license. For more details, refer to the `LICENSE` document in the repository.

---

## 7. Additional Notes
- Ensure that all required Python libraries are installed for successful execution.
- For accurate predictions, it's recommended to regularly update the dataset with recent sales and inventory data.
