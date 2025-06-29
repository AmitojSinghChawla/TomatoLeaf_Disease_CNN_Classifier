from flask import Flask, render_template, request
from utils import load_model, predict_image

app = Flask(__name__)
model = load_model()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        if "image" in request.files:
            image = request.files["image"]
            if image.filename != "":
                prediction = predict_image(image, model)
    return render_template("index.html", prediction=prediction)
