# Estimating Carbon Emissions of HuggingFace AI Models
This repository contains the code for analyzing CO2 emissions data from Hugging Face models, as well as a Firefox extension for predicting carbon emissions of Autotrain Hugging Face models. The analysis is conducted using `statistics.py` to gather dataset statistics and `ml_models.py` to perform linear regression and random forest regression on the dataset.

## Repository Structure
`statistics.py`: Python script for gathering statistics of the dataset.  
`ml_models.py`: Python script for performing linear regression and random forest regression on the dataset.  
Ecolabel/: Folder containing the code for the Firefox extension.  
Python_server/: Folder containing the Flask backend code for the Firefox extension.  

## Installation and Usage
### Analysis
`statistics.py` and `ml_models.py` can be ran as is with `python {file}.py`.

### Firefox Extension
1. Load the extension into Firefox:
    - Open Firefox and type about:debugging in the address bar.
    - Click on "This Firefox" in the left-hand menu.
    - Click on "Load Temporary Add-on" and navigate to the manifest.json file in the Ecolabel/ folder.
    - Select the manifest.json file to load the extension.
2. Once the extension is loaded, you can use it by following these steps:
    - Click on the extension icon in the toolbar.
    - Select the domain of your machine learning model (NLP, Computer Vision, or Other).
    - Enter the size of your dataset in bytes.
    - The extension will provide you with a prediction of carbon emissions in grams.

### Flask server
Is hosted and referred to by the extension, but the code could be hosted from `flask_app.py`. The extension will need adjusted routing in that case.

## Contributors
Thijs Nulle ([@thijsnulle](https://github.com/thijsnulle))  
Petter Reijalt ([@Petter6](https://github.com/Petter6))  
Harmen Kroon ([@HarmenKroon](https://github.com/HarmenKroon))  
