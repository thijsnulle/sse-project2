from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pickle
import os
import requests
import colourMap as cmap

app = Flask(__name__)
CORS(app)

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Path to your .pkl file
pkl_file_path = os.path.join(current_dir, 'linear_regression_model.pkl')

model_dict={}

# Load the linear regression model from the .pkl file
with open(pkl_file_path, 'rb') as f:
    model_dict = pickle.load(f)

coefficients = model_dict['coefficients']
intercept = model_dict['intercept']
feature_names = model_dict['feature_names']

# Your existing route
@app.route('/')
def index():
    return 'Hello, World!'

# Define an API endpoint for prediction
@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    # Check if it's a preflight request
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
    else:
        # Get data from request
        data = request.get_json()

        # Extract features from data
        datasets_size = data['datasetSize']
        feature = data['domain']

        # Prepare features for prediction
        # Create a 2D array with one row and the same number of columns as the coefficients
        # Fill in the values for the provided features, and set the rest to 0
        X_test = np.zeros((1, len(coefficients)))
        X_test[0, feature_names.get_loc('datasets_size')] = datasets_size
        X_test[0, feature_names.get_loc(f'domain_{feature}')] = 1
        X_test[0, feature_names.get_loc('auto')] = 1

        # Compute prediction using only the relevant coefficients
        prediction = np.dot(X_test, coefficients) + intercept

        # Compute prediction colour
        prediction_colour = cmap.colourMap(prediction)

        # Return the prediction as JSON response
        response = jsonify({'prediction': prediction.tolist(), 'colour_prediction': prediction_colour})

    # Set CORS headers
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'

    return response

@app.route('/proxy', methods=['POST'])
def proxy():
    url = request.json.get('url')  # Get the URL to proxy from the query parameters
    try:
        # Fetch the content of the URL using requests
        response = requests.get(url)

        # Return the content to the frontend
        return jsonify({'content': response.text})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)