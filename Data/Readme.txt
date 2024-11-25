README.txt - Data Folder

This folder contains all data files utilized in the Structured Data Project.

### Data Overview:
The data used in this pipeline pertains to air quality measurements.

**Data Source:**  https://archive.ics.uci.edu/dataset/360/air+quality

### Files:

#### 1. Raw Data
- **AirQualityUCI.xlsx** - Raw data downloaded from the source in Excel format.
- **AirQualityUCI.csv** - Raw data downloaded from the source in CSV format.

#### 2. Preprocessed Data
- **UpdateD_AirQualityUCI.xlsx** - Preprocessed data saved from the preprocessing pipeline in Excel format.
- **UpdateD_AirQualityUCI.csv** - Preprocessed data saved from the preprocessing pipeline in CSV format.

#### 3. Train and Test Data
- **var_log_train_data.csv** - Contains 80% of the logged data from top correlated features with the target variable, generated for training the VAR model.
- **var_log_test_data.csv** - Contains 20% of the logged data from top correlated features with the target variable, generated for testing the VAR model.
- **var_sqrt_train_data.csv** - Contains 80% of square-root-transformed (power-transformed) data from top correlated features with the target variable, generated for training the VAR model.
- **var_sqrt_test_data.csv** - Contains 20% of square-root-transformed (power-transformed) data from top correlated features with the target variable, generated for testing the VAR model.
- **xgboost_train_data.csv** - Contains 80% of the normalized data from top correlated features with the target variable, generated for training the XGBoost model.
- **xgboost_test_data.csv** - Contains 20% of the normalized data from top correlated features with the target variable, generated for testing the XGBoost model.


Each file within this folder serves a specific purpose within the project pipeline, from raw data acquisition to preprocessed transformations, and train-test splits for model building.
