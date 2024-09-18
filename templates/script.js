// Get references to form elements
const form = document.getElementById('commentForm');
const clearBtn = document.getElementById('clearBtn');
const commentArea = document.getElementById('commentArea');
const resultDiv = document.getElementById('predictionResult');

// Clear textarea when clear button is clicked
clearBtn.addEventListener('click', function() {
    commentArea.value = '';
    resultDiv.textContent = '';
});

// Handle form submission with prediction result display
form.addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent default form submission

    // Simulate prediction result (replace with actual prediction logic)
    const prediction = "Cyberbullying Detected!";
    resultDiv.textContent = prediction;

    // You can also send the form data via AJAX for dynamic submission if needed
    // Example: Use fetch() for form submission to the backend
    /*
    fetch('/predict', {
        method: 'POST',
        body: new FormData(form)
    })
    .then(response => response.json())
    .then(data => {
        resultDiv.textContent = data.prediction;  // Update with actual result from the backend
    })
    .catch(error => console.error('Error:', error));
    */
});
