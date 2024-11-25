# README.txt - XGBoost (Extreme Gradient Boosting)

## Overview
XGBoost (Extreme Gadient Boosting) is ensemble of decision tree used to analyze and predict multiple sequential variables, capturing the relationships between them over time.

Since our dataset involves time-series data which is sequential form of dataset, also as it does not have any seasonality we selected the XGBoost model due to its ability to dynamically predicting the future values of the data.

### Files:
- **Utilities.py**  
  Contains function definitions for training, testing, and evaluating the XGBoost model.

- **XGBoost_model_training_and_evaluation.ipynb**  
  Jupyter notebook implementing the training and testing of the XGBoost model, with results.

- **xgboost_model.h5**  
  Saved model trained on normalized/scaled data.

The XGBoost model performs well for our non-Gaussian data (i.e. close fit to the actual data).
