import scipy.stats as stats
import numpy as np
import os

def colourMap(prediction):
	# Get the current directory
	current_dir = os.path.dirname(os.path.abspath(__file__))

	# Path to sorted emissions file
	emissions_file_path = os.path.join(current_dir, 'sorted_auto_carbon_emissions')

	sorted_auto_carbon_emissions_grams = np.fromfile(emissions_file_path)

	colours = ['green', 'yellow', 'orange', 'red']
	colour_prediction = 'blue'
	if (stats.percentileofscore(sorted_auto_carbon_emissions_grams, prediction) <= 25) :
		colour_prediction = colours[0]
	elif (stats.percentileofscore(sorted_auto_carbon_emissions_grams, prediction) <= 50) :
		colour_prediction = colours[1]
	elif (stats.percentileofscore(sorted_auto_carbon_emissions_grams, prediction) <= 75):
		colour_prediction = colours[2]
	else:
		colour_prediction = colours[3]

	return colour_prediction