from utils import load_data, preprocess_data, train_test_split_data
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
import numpy as np
import matplotlib.pyplot as plt


def perform_linear_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    if np.all(y_pred >= 0):
        log_mse = mean_squared_log_error(y_test, y_pred)
    else:
        log_mse = 'N/A'
    r2 = r2_score(y_test, y_pred)

    print("Linear Regression Results:")
    print(f"MSE: {mse}, LogMSE: {log_mse}, R^2: {r2}")

    plt.scatter(y_test, y_pred)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("True vs. Predicted Values")
    plt.show()

def perform_random_forest_regression(X, y, n=5):
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    model = RandomForestRegressor(n_estimators=n, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    if np.all(y_pred >= 0):
        log_mse = mean_squared_log_error(y_test, y_pred)
    else:
        log_mse = 'N/A'
    r2 = r2_score(y_test, y_pred)

    print(f"Random Forest Regression ({n=}) Results:")
    print(f"MSE: {mse}, LogMSE: {log_mse}, R^2: {r2}")

    plt.scatter(y_test, y_pred)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Random Forest Regression ({n=})")
    plt.show()

def perform_svm_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    model = SVR(kernel='linear')  # You can choose different kernel functions (linear, polynomial, etc.)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    if np.all(y_pred >= 0):
        log_mse = mean_squared_log_error(y_test, y_pred)
    else:
        log_mse = 'N/A'
    r2 = r2_score(y_test, y_pred)

    print("Support Vector Machine Regression Results:")
    print(f"MSE: {mse}, LogMSE: {log_mse}, R^2: {r2}")

    plt.scatter(y_test, y_pred)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Support Vector Machine Regression")
    plt.show()

def perform_regression(file_path, input, method='linear_regression'):
    X, y = load_data(file_path, input=input)
    X = preprocess_data(X)

    if method == 'linear':
        perform_linear_regression(X, y)
    elif method == 'random_forest':
        perform_random_forest_regression(X, y, n=5)
    elif method == 'svm':
        perform_svm_regression(X, y)
    else:
        print("Invalid method. Choose 'random_forest' or 'svm'.")

if __name__ == '__main__':
    small = ['datasets_size', 'size']
    medium = ['datasets_size', 'size', 'geographical_location', 'hardware_used']
    perform_regression('../datasets/HFClean.csv', input=medium, method='random_forest')
