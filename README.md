# Vending Machine Sales Forecasting

This repository provides solutions for forecasting sales of products in a vending machine using two different approaches: AdaBoost and LSTM.

## Directory Structure

    .
    ├── adaboost
    │   ├── dataset
    │   │   ├── product1.csv
    │   │   └── product2.csv
    │   ├── training.py
    │   └── inference.py
    ├── lstm
    │   ├── dataset
    │   │   ├── product1.csv
    │   │   └── product2.csv
    │   ├── training.py
    │   └── inference.py
    └── LICENSE


## 📄 Dataset

Each product dataset spans 60 days and features the following columns:
- **Date**: The date of sales.
- **Sales**: The number of sales on that date.

Additional data attributes like holidays and weekends have been extrapolated based on the date.

## ⚙️ Usage

Both the AdaBoost and LSTM directories share a similar structure and usage pattern.

### Training

To train a model, navigate to either the `AdaBoost` or `LSTM` directory and run:

```bash
python training.py 
python inference.py --date <date till which to forecast sales>
```
## License
This project is licensed under the terms mentioned in the LICENSE file.
