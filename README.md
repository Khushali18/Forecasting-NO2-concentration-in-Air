# README.txt

## Aim
The goal of this project is to predict NO2 concentration in the air over time.

To develop this predictive pipeline, we require an air quality dataset containing ground truth values of gas concentrations, along with sensor records. We aim to predict sensor outputs over time. We have obtained a dataset with concentrations of five gases in the air from an Italian city, comprising nearly 10,000 records. This raw data is located in the **Data** folder.

As we know, the quality of data is crucial for any machine learning model. By providing high-quality data that is understandable by the model or algorithm, we can achieve optimal performance. Therefore, we must analyze and preprocess this data. The implementation of these processes can be found in the **Preprocessing** folder.

With the preprocessed data ready, we can input it into the models most suitable for our dataset. Based on our analysis during the preprocessing task, we have selected the VAR and XGBoost models. The implementation of these models can be found in the **Models** folder.

The **Raw_Pipeline.ipynb** file contains the complete implementation, covering everything from data exploration and preprocessing to model training and evaluation.

The **Plan.docx** outlines the action plan for the project.

## Folder Structure

Structure_Data_Pipeline<br>
 |
 |---Data
 |
 |---Models
 |     |
 |     |---VAR
 |     |
 |     |---XGBoost
 |
 |---Preprocessing
