import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

def parse_date_time(df, index):
    """
    Function to convert datatypes of Date & Time columns to timestamp and combine them into one single column
    Args:
        df(pd.DataFrame): Dataframe of the data to be parsed
        index(int): index to add DateTime column in Dataframe
    Returns:
        df(pd.DataFrame): Updated Dataframe
    """
    #parse Dat and Time to DateTime
    df['Date'] = pd.to_datetime(df['Date'])
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time
    df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))

    # Setting DateTime Column at Particular position in columns list
    columns = df.columns.tolist()
    columns.insert(index, columns.pop(columns.index('DateTime')))
    df = df[columns]
    df.set_index('DateTime', inplace=True)
    df = df.drop(columns = ['Date', 'Time'])
    return df



def handle_missing_values(df, method='median'):
    """
    Function to check missing values in data and impute them
    Args:
        df(pandas.DataFrame): Dataframe of the data to be checked
        method(String): method using which we want to fill the NAN values if any
    Returns:
        df(pandas.Dataframe): Updated Dataframe
    
    Note: Here, in this data the NAN values are already been imputed by value -200
    """
    # Checking missing values by variable
    print(f'{"No missing values" if df.isnull().sum().sum()==0 else f"Handling missing values with {method} values"}')
    print(df.isnull().sum())
    # print(df[df['AH']==-200])
    # print((df['AH']==-200).sum())

    # Substituting NAN values with -200
    if df.select_dtypes(include="number").isnull().sum().sum() != 0:
        numeric_df = df.select_dtypes(include="number")
        if method == 'median':
            df.fillna(df.median(), inplace=True)
        elif method == 'mean':
            df.fillna(df.mean(), inplace=True)
        elif method == 'ffill':
            df.fillna(method='ffill', inplace=True)   # Forward fill
        elif method == 'bfill':
            df.fillna(method='bfill', inplace=True)  # backward fill
    return df



def generate_line_plots(df, columns, start_timespan='2004-03-01 00:00:00', end_timespan='2005-04-05 00:00:00'):
    """
    Function to plot trend and seasonality in our time-series data
    Args:
        df(pandas.Dataframe): Dataframe of the loaded data
        columns(List): list of colimns for which we want to plot line plots
        start_timespan(String): Start date and time if want to see plots for specific duration(Default: '2004-03-01 00:00:00')
        end_timespan(String): End date and time for that specific time duration(Default: '2005-04-05 00:00:00')
    Returns:
        None
    Note: Displays line plots of all specified columns
    """
    start_timespan = pd.to_datetime(start_timespan)
    end_timespan = pd.to_datetime(end_timespan)
    
    sns.set(style='whitegrid')
    plt.figure(figsize=(15, 20))
    for i, column in enumerate(columns, 1):
        plt.subplot(len(columns), 1, i)
        plt.plot(df.index, df[column], label=column, color='blue')
        plt.title(f'Line Plot of {column}')
        plt.xlabel('DateTime')
        plt.xlim(start_timespan, end_timespan)
        plt.ylabel(column)
        plt.xticks(rotation=45)
        plt.legend()
        
    plt.tight_layout()
    plt.show()
    return None



def decompose_seasonality_charts(df, columns):
    """
    Function to plot trend, seasonality in our time-series data
    Args:
        df(pandas.Dataframe): Dataframe of the loaded data
        columns(List): list of colimns for which we want to plot line plots
    Returns:
        None
    Note: Displays trend, seasonality and residuals plots of all specified columns
    """
    sns.set(style='whitegrid')
    plt.figure(figsize=(15, 20))
    # Analyze trend and seasonality for each column
    for i, column in enumerate(columns, 1):
        plt.subplot(len(columns), 1, i)
        decomposition = seasonal_decompose(df[column], model='additive', period=24)  # Since we have daily data with hourly frequency
        decomposition.plot()
        # plt.title(f'Trend and Seasonality Decomposition for {column}')
        
    plt.tight_layout()
    plt.show()
    return None



def check_uniformity(df, columns):
    """
    Function to check the uniformity of numerical data and print the results
    Args:
        df(pandas.DataFrame): Dataframe of the data
        columns(List): List of columns to be checked
    Returns:
        None
    Note: It displays the uniformity plots for specified columns
    """
    uniform = []
    non_uniform = []
    plt.figure(figsize=(15, 20))
    for i, column in enumerate(columns, 1):
        plt.subplot(len(columns), 1, i)
        sns.histplot(df[column], bins=30, kde=True)
        
        # Kolmogorov-Smirnov test for uniform distribution
        data = df[column].dropna()
        k_statistic, p_value = stats.kstest(data, 'uniform', args=(data.min(), data.max() - data.min()))
        plt.title(f'Distribution of {column} - K-S Test: Statistic={k_statistic:.3f}, p-value={p_value:.3f}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        
        # Check if the p-value is less than the significance level (alpha = 0.05)
        if p_value < 0.05:
            plt.axvline(data.mean(), color='red', linestyle='dashed', linewidth=1)
            plt.text(data.mean(), plt.ylim()[1]*0.9, 'Mean', color='red', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    for column in columns:
        data = df[column].dropna()
        k_statistic, p_value = stats.kstest(data, 'uniform', args=(data.min(), data.max() - data.min()))
        # print(f'{column} - K-S Test: Statistic={k_statistic:.3f}, p-value={p_value:.3f} - {"Uniform" if p_value >= 0.05 else "Not Uniform"}')
        if p_value >= 0.05:
            uniform.append(column)
        else:
            non_uniform.append(column)
    if len(uniform) > 0:
        print(f'{uniform} are Uniform')
    if len(non_uniform) > 0:
        print(f'{non_uniform} are Non-Uniform')
    return None



def replace_negative_values_with_nan(df):
    """
    Function to replace -200 values with nan, as it can affect the uniformity of the data
    Args:
        df(pandas.DataFrame): Dataframe of the loaded data
    Returns:
        df(pandas.DataFrame): updated dataframe
    """
    df.replace(-200, np.nan, inplace=True)
    return df



def normalize_or_scale_data(df):
    """
    Function to sclae data using MinMaxscaler
    Args:
        df(pandas.DataFrame): Dataframe of the data
    Returns:
        df(pandas.DataFrame): updated dataframe
    """
    # Normalize data using min_max scaler
    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])
    return df



def detect_outliers_iqr(data):
    """
    Function to detect outliers using IQR method
    Args:
        data(List): list of all values of a column
    Returns:
        outliers(List): list of all extreme values or outliers
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)))
    return outliers



def handle_outliers(df):
    """
    Function to handle outliers, i.e. replace the outlier values with median values
    Args:
        df(pandas.DataFrame): Dataframe of the data
    Returns:
        df(pandas.DataFrame): Updated dataframe
    """
    for col in df.columns:
        outliers = detect_outliers_iqr(df[col])
        df.loc[outliers, col] = df[col].median()
    return df


def correlation_analysis(df):
    """
    Function for correlation analysis
    Args:
        df(pandas.DataFrame): Dataframe of loaded data
    Returns:
        correlation_matrix(pandas.Dataframe): correlation matrix of the features 
    Note: Displays Correlation Matrix for features
    """
    correlation_matrix = df.corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()
    return correlation_matrix



def top_correlated_features(df, correlation_matrix, target_variable, number):
    """
    Function to select correlated features for target variable from top correlated features and plot their visualizations
    Args:
	df(pandas.DataFrame): Dataframe of the loaded data
        correlation_matrix(pandas.Dataframe): correlation matrix of the features
        target_variable(String): target variable fortop correlated features
        number(Integer): number of top correlated features required
    Returns:
        None
    """
    target_corr = correlation_matrix[target_variable].sort_values(ascending=False)
    print(f"Top features correlated with {target_variable}:")
    print(target_corr.head(number))
    # Visualizations and analysis for forecasting readiness of top correlated features
    sns.pairplot(df, vars=target_corr.head(5).index)
    plt.show()
    return None


def log_transform(df, column, index):
    """
    Function to log transform for the variable
    Args:
        df(pandas.DataFrame): Dataframe of our data
        column(String): column name to get transformed
        index(Integer): index at which column need to be added in dataframe
    Returns:
        df(pandas.DataFrame): updated dataframe
    """
    df['log_'+column] = np.log1p(df[column])
    columns = df.columns.tolist()
    columns.insert(index, columns.pop(columns.index('log_'+column)))
    df = df[columns]
    return df



def sqrt_transformation(df, column, index):
    """
    Function to perform square root transform for the variable
    Args:
        df(pandas.DataFrame): Dataframe of our data
        column(String): column name to get transformed
        index(Integer): index at which column need to be added in dataframe
    Returns:
        df(pandas.DataFrame): updated dataframe
    """
    df['sqrt_'+column] = np.sqrt(df[column])
    columns = df.columns.tolist()
    columns.insert(index, columns.pop(columns.index('sqrt_'+column)))
    df = df[columns]
    return df
