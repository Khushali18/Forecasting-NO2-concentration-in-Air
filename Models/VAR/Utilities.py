import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle



def train_and_save_model(train_data, target_columns, model_path="var_model.pkl"):
    """
    Function to train model with VAR algorithm
    Args:
        train_data(pandas.DataFrame): training dataframe from the data of selected features
        target_columns(List): columns to be used for forecasting
        model_path(String): Path to save the model
    Returns:
        var_model(statsmodel): trained VAR model
    """
    model = VAR(train_data[target_columns])
    var_model = model.fit()
    
    # Save trained model
    with open(model_path, "wb") as file:
        pickle.dump(var_model, file)
    
    print(f"Model trained and saved at {model_path}")
    return var_model



def load_and_test_model(test_data, target_column, model_path="var_model.pkl"):
    """
    Function to test model
    Args:
        test_data(pandas.DataFrame): testing dataframe from the data of selected features
        target_column(string): target column one to be predicted
        model_path(String): Path to the saved trained model
    Returns:
        forecast_df(pandas.Dataframe): forecasted values of target variable in dataframe
        eval_dict(Dictionary): Evaluation metric values
    """
    # Load model
    with open(model_path, "rb") as file:
        var_model = pickle.load(file)
    
    # predictions
    lag_order = var_model.k_ar
    test_data_lagged = test_data.values[-lag_order:]
    forecast = var_model.forecast(test_data_lagged, steps=len(test_data))
    forecast_df = pd.DataFrame(forecast, index=test_data.index, columns=target_column)
    
    # Evaluate model performance
    rmse = mean_squared_error(test_data[target_column], forecast_df, squared=False)
    mae = mean_absolute_error(test_data[target_column], forecast_df)

    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")

    eval_dict = {"RMSE": rmse, "MAE": mae}

    return forecast_df, eval_dict

