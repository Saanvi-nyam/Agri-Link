let model;

// Load the TensorFlow.js model
async function loadModel() {
    try {
        model = await tf.loadLayersModel('model.json'); // Update with relative path
        console.log("Model loaded.");
    } catch (error) {
        console.error("Error loading model:", error);
    }
}

// Function to display the selected image

// Function to make a prediction
async function predict() {
    if (!model) {
        alert("Model is not loaded yet. Please wait.");
        return;
    }

    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];

    if (!file) {
        alert("Please select an image file.");
        return;
    }

    console.log("Selected file:", file);
    const img = document.createElement('img');
    img.src = URL.createObjectURL(file);
    img.onload = async () => {
        const tensorImg = tf.browser.fromPixels(img)
            .resizeNearestNeighbor([128, 128]) // Resize to match model's input shape
            .toFloat()
            .expandDims()
            .div(tf.scalar(255)); // Normalize to [0, 1]

        console.log("Preprocessed image shape:", tensorImg.shape);

        const prediction = model.predict(tensorImg);
        console.log("Raw prediction result:", prediction);

        const probabilities = prediction.softmax();
        probabilities.print(); // Log the probabilities

        const classIndex = probabilities.argMax(-1).dataSync()[0];
        console.log(`Predicted Class Index: ${classIndex}`);

        const classesResponse = await fetch('classes.json'); // Update with relative path
        if (!classesResponse.ok) {
            console.error("Failed to load classes.json:", classesResponse.statusText);
            return;
        }

        const classMap = await classesResponse.json();
        console.log("Class mapping loaded:", classMap);

        // Convert object to array
        const classNames = Object.keys(classMap);
        console.log("Class names array:", classNames);

        // Check if classIndex is valid
        if (classIndex < 0 || classIndex >= classNames.length) {
            console.error("Invalid class index:", classIndex);
            document.getElementById('result').innerText = "Prediction failed.";
            return;
        }

        document.getElementById('result').innerText = `Predicted Class: ${classNames[classIndex]}`;
        document.getElementById('predictedImage').src = img.src;
        document.getElementById('predictedImage').style.display = 'block';
    };

    img.onerror = () => {
        console.error("Error loading the image.");
        alert("Error loading the image. Please try again.");
    };
}

document.getElementById('predictButton').addEventListener('click', predict);

// Load the model when the page loads
loadModel();
