document.addEventListener('DOMContentLoaded', function() {
  const predictBtn = document.getElementById('predict-btn');
  const resultDiv = document.getElementById('result');
  const cropResult = document.getElementById('crop-result');
  const errorDiv = document.getElementById('error');
  const errorMessage = document.querySelector('.error-message');
  
  // API endpoint (you'll need to host your Flask app somewhere)
  const API_URL = 'https://your-api-endpoint.com/predict';
  
  predictBtn.addEventListener('click', function() {
    // Get input values
    const n = parseFloat(document.getElementById('n').value);
    const p = parseFloat(document.getElementById('p').value);
    const k = parseFloat(document.getElementById('k').value);
    const temperature = parseFloat(document.getElementById('temperature').value);
    const humidity = parseFloat(document.getElementById('humidity').value);
    const ph = parseFloat(document.getElementById('ph').value);
    const rainfall = parseFloat(document.getElementById('rainfall').value);
    
    // Validate inputs
    if (isNaN(n) || isNaN(p) || isNaN(k) || isNaN(temperature) || 
        isNaN(humidity) || isNaN(ph) || isNaN(rainfall)) {
      showError('Please fill in all fields with valid numbers.');
      return;
    }
    
    // Prepare data for API
    const data = {
      n: n,
      p: p,
      k: k,
      temperature: temperature,
      humidity: humidity,
      ph: ph,
      rainfall: rainfall
    };
    
    // Show loading state
    predictBtn.textContent = 'Loading...';
    predictBtn.disabled = true;
    
    // Call API
    fetch(API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
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
      if (data.error) {
        showError(data.error);
      } else {
        // Show result
        cropResult.textContent = data.crop;
        resultDiv.classList.remove('hidden');
        errorDiv.classList.add('hidden');
      }
    })
    .catch(error => {
      showError('Error: ' + error.message);
    })
    .finally(() => {
      // Reset button
      predictBtn.textContent = 'Get Recommendation';
      predictBtn.disabled = false;
    });
  });
  
  function showError(message) {
    errorMessage.textContent = message;
    errorDiv.classList.remove('hidden');
    resultDiv.classList.add('hidden');
  }
});
