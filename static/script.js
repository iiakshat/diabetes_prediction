document.getElementById('prediction-form').addEventListener('submit', async function(event) {
    event.preventDefault();

    const formData = new FormData(event.target);
    const formObject = {};
    formData.forEach((value, key) => formObject[key] = value);

    try {
        const response = await fetch('/predictions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formObject)
        });

        if (response.ok) {
            const result = await response.json();
            document.getElementById('result').textContent = `Predicted Academic Success Score: ${result.predicted_academic_success_score}`;
        } else {
            document.getElementById('result').textContent = 'Error in prediction';
        }
    } catch (error) {
        document.getElementById('result').textContent = 'An error occurred';
    }
});
