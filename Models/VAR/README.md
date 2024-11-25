# README.txt - Vector Autoregression (VAR)

## Overview
Vector Autoregression (VAR) is a statistical model used to analyze and predict multiple time-dependent variables, capturing the relationships between them over time.

Since our dataset involves time-series data with multiple correlated features, we selected the VAR model due to its ability to dynamically model these relationships and leverage the dependencies between feature variables for forecasting.

### Files:
- **Utilities.py**  
  Contains function definitions for training, testing, and evaluating the VAR model.

- **VAR_model_training_and_evaluation.ipynb**  
  Jupyter notebook implementing the training and testing of the VAR model, with results.

- **var_model_log.pkl**  
  Saved model trained on logged data.

- **var_model_sqrt.pkl**  
  Saved model trained on square root transformed data.

The VAR model performs adequately for our non-Gaussian data.
