# Bike Sharing Demand Prediction

A machine-learning project for predicting bike-sharing demand using Python, Pandas, NumPy, exploratory data analysis, feature engineering, and Gradient Boosting Regression.

## Project Overview

This project analyzes bike-sharing demand data, extracts time-based features from the datetime information, explores demand patterns, trains a Gradient Boosting regression model, evaluates predictions using RMSLE, and generates a Kaggle-compatible submission file.

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- GradientBoostingRegressor

## Project Workflow

1. Load the bike-sharing dataset.
2. Inspect and preprocess the data.
3. Parse the datetime information.
4. Create time-based features.
5. Perform exploratory data analysis.
6. Apply target transformation.
7. Select model features.
8. Split the data into training and validation sets.
9. Train a Gradient Boosting Regression model.
10. Evaluate the model using RMSLE.
11. Analyze feature importance.
12. Generate predictions for the test data.
13. Create a `submission.csv` file for the competition workflow.

## Feature Engineering

The notebook creates time-related features from the datetime column, including:

- Year
- Month
- Day
- Hour
- Day of week
- Month name

The analysis also examines demand patterns across different hours, days, and weather conditions.

## Exploratory Data Analysis

The notebook uses Matplotlib and Seaborn to investigate relationships between bike demand and temporal/environmental variables.

Examples include:

- Demand by hour
- Demand by day of week
- Demand by weather condition
- Feature relationships and distributions

## Machine Learning

### Model

**GradientBoostingRegressor**

The model is trained using the engineered features and evaluated on a validation split.

### Evaluation

The project uses **Root Mean Squared Logarithmic Error (RMSLE)** as the main evaluation metric.

RMSLE is appropriate for demand-prediction tasks where relative differences between predicted and actual demand are important.

## Feature Importance

The trained Gradient Boosting model is used to inspect feature importance and understand which engineered variables contribute most to the prediction.

## Predictions and Submission

The notebook generates predictions for the test data and creates a `submission.csv` file in the format required for the competition workflow.

## Repository Structure

```text
bike-sharing-demand-prediction/
├── README.md
├── bike_sharing_demand.py
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── Bike_sharing_Demand_Forecasting.ipynb
├── models/
│   └── README.md
└── visualizations/
    └── README.md
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

For the complete experiment, open:

```text
notebooks/Bike_sharing_Demand_Forecasting.ipynb
```

in Jupyter Notebook or Google Colab and run the notebook cells.

## Notes

This repository reflects the workflow implemented in the original project notebook. It does not claim a specialized time-series forecasting algorithm; the notebook uses a regression approach with engineered time-based features and a train/validation split.

## Author

**Mohammad Rashid**

B.Tech in Artificial Intelligence and Data Science
