import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path, input=[]):
    """
    Load data from a CSV file and return the features (X) and target (y).
    """
    data = pd.read_csv(file_path)
    X = data[input]  # Replace 'target_column' with the name of your target column
    y = data['co2_eq_emissions']  # Replace 'target_column' with the name of your target column
    return X, y

def preprocess_data(X):
    """
    Preprocess the data by scaling features using MinMaxScaler.
    """
    X = pd.get_dummies(data=X, drop_first=True)
    scaler = MinMaxScaler(feature_range=(-1,1)).fit(X)
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def preprocess_missing_data(data):
    for column in data.columns:
        if column in ['parameters']: # Add all columns of data to be flagged
            missing_values = data[column].isnull()  # Check for missing values in the column
            flag_column_name = column + '_missing_flag'  # Name for the new flag column
            data[flag_column_name] = missing_values.astype(int)  # Create a new flag column with 1s for missing values and 0s otherwise
    return data

def train_test_split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
