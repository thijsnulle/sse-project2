document.addEventListener("DOMContentLoaded", function() {
  const submitBtn = document.getElementById("submitBtn");
  
  const domainSelect = document.getElementById("domain");
  const datasetSizeInput = document.getElementById("datasetSize");

  // Load the inputs from the browser's local storage if available
  if (localStorage.getItem("domain")) {
    domainSelect.value = localStorage.getItem("domain");
  }
  if (localStorage.getItem("datasetSize")) {
    datasetSizeInput.value = localStorage.getItem("datasetSize");
  }
  
  submitBtn.addEventListener("click", function() {
    const domain = document.getElementById("domain").value;
    const datasetSize = document.getElementById("datasetSize").value;

	// Store the inputs in the browser's local storage
    localStorage.setItem("domain", domain);
    localStorage.setItem("datasetSize", datasetSize);
	
    // Construct the data object
    const data = {
      domain: domain,
      datasetSize: datasetSize
    };
	
	console.log('json package:', data);

    // Send the data to the backend
    fetch('https://harmenk.pythonanywhere.com/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(data)
    })
    .then(response => {
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      return response.json();
    })
    .then(data => {
      console.log('Response from server:', data);
      // Once response is received, render the prediction result on the webpage
	  renderPrediction(data.prediction, data.colour_prediction);
    })
    .catch(error => {
      console.error('There was a problem with the fetch operation:', error);
    });
  });
  
  // Function to render the prediction in the popup
  function renderPrediction(prediction, colour) {
    const predictionElement = document.getElementById("prediction");
    predictionElement.textContent = "Prediction: " + prediction + " grams";
    predictionElement.style.color = colour;
  }
});
