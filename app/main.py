# main.py or app.py

# Importing required modules from Flask and others
from flask import Flask, request, render_template   # Flask app, form data handling, and HTML rendering
from io import BytesIO                              # To handle in-memory file objects
from PIL import Image                               # To open and process image files
import base64                                       # For encoding image to display in HTML

# Importing custom utility functions and variables from app/utils.py
from app.utils import load_model, predict_image, num_classes

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model once when the app starts, using the number of classes
model = load_model(num_classes=num_classes)

# Define a route for the main page ("/") that accepts both GET and POST methods
@app.route("/", methods=["GET", "POST"])
def index():
    # Initialize variables to None; these will be passed to the HTML template
    prediction = None
    confidence = None
    description = None
    treatment = None
    image_base64 = None

    # Handle POST request (form/image submission)
    if request.method == "POST":
        # Check if an image file was included in the form
        if "image" not in request.files:
            prediction = "⚠️ No file part in request"
        else:
            file = request.files["image"]
            # Check if a file was selected (filename not empty)
            if file.filename == "":
                prediction = "⚠️ No selected file"
            else:
                try:
                    # Read and decode the uploaded image file
                    image_bytes = file.read()
                    image = Image.open(BytesIO(image_bytes)).convert("RGB")

                    # Make prediction using the loaded model
                    result = predict_image(image, model)
                    prediction = result["class"]           # Predicted class
                    confidence = result["confidence"]      # Confidence score
                    description = result["description"]    # Class description
                    treatment = result["treatment"]        # Suggested treatment or info

                    # Encode image to base64 so it can be displayed in the web page
                    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

                # Catch and report any error that occurred during processing
                except Exception as e:
                    prediction = f"❌ Error: {str(e)}"
                    confidence = None
                    description = None
                    treatment = None

    # Render the HTML template with the result data
    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        description=description,
        treatment=treatment,
        image_data=image_base64
    )
