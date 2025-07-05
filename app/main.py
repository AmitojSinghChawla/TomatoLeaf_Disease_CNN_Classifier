# main.py or app.py
from flask import Flask, request, render_template
from io import BytesIO
from PIL import Image
import base64

from app.utils import load_model, predict_image, num_classes

app = Flask(__name__)

# Load model once at startup
model = load_model(num_classes=num_classes)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    description = None
    treatment = None
    image_base64 = None

    if request.method == "POST":
        if "image" not in request.files:
            prediction = "⚠️ No file part in request"
        else:
            file = request.files["image"]
            if file.filename == "":
                prediction = "⚠️ No selected file"
            else:
                try:
                    image_bytes = file.read()
                    image = Image.open(BytesIO(image_bytes)).convert("RGB")
                    result = predict_image(image, model)
                    prediction = result["class"]
                    confidence = result["confidence"]
                    description = result["description"]
                    treatment = result["treatment"]

                    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                except Exception as e:
                    prediction = f"❌ Error: {str(e)}"
                    confidence = None
                    description = None
                    treatment = None

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        description=description,
        treatment=treatment,
        image_data=image_base64
    )

if __name__ == "__main__":
    app.run(debug=True)
