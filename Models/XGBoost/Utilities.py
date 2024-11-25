import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt


def create_sequences(data, seq_length=5):
    """
    Function to prepare sequences for XGBoost input (flattened format).
    Args:
        data (pandas.Dataframe): DataFrame of loaded data
        seq_length (int): time steps for each sequence
    Returns:
        X (numpy array): input sequence data, flattened
        y (numpy array): target variable sequence
    """
    if isinstance(data, pd.DataFrame):
        data = data.values

    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length].flatten())  
        y.append(data[i + seq_length, 0])  
    
    return np.array(X), np.array(y)
	

def train_and_save_model(X_train, y_train, model_path="xgboost_model.pkl"):
    """
    Function to train model with XGBoost algorithm
    Args:
        X_train (numpy array): Training feature data of selected features
        y_train (numpy array): Training target data
        model_path (str): Path to save the model
    Returns:
        xgb_model (Booster): Trained XGBoost model
    """
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    # Train model
    xgb_model.fit(X_train, y_train)
    
    # Save model
    with open(model_path, "wb") as model_file:
        pickle.dump(xgb_model, model_file)
    print(f"Model trained and saved at {model_path}")
    return xgb_model



def load_and_test_model(X_test, y_test, model_path="xgboost_model.pkl"):
    """
    Function to test XGBoost model
    Args:
        X_test (numpy array): Testing feature data
        y_test (numpy array): True target values
        model_path (str): Path to the saved trained model
    Returns:
        predictions (numpy array): Predicted values
        eval_dict (dict): Evaluation metric values
    """
    # Load model
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)

    # Predictions
    predictions = model.predict(X_test)
    
    # Evaluation
    rmse = mean_squared_error(y_test, predictions, squared=False)
    mae = mean_absolute_error(y_test, predictions)

    eval_dict = {"RMSE": rmse, "MAE": mae}
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    return predictions, eval_dict

