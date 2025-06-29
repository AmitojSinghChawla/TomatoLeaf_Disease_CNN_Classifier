import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from app.utils import load_model, predict_image

# --- Config ---
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# --- Flask App Setup ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Load Model Once at Startup ---
model = load_model()

# --- Utility: Check allowed file extensions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Utility: Clear previous uploaded images ---
def clear_upload_folder():
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"⚠️ Error deleting file {file_path}: {e}")

# --- Main Route ---
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_url = None

    if request.method == 'POST':
        file = request.files['image']
        if file and allowed_file(file.filename):
            # Clean up old images
            clear_upload_folder()

            # Save new image
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Predict using model
            prediction = predict_image(filepath, model)
            image_url = filepath  # Path used in <img src="{{ image_url }}">

    return render_template('index.html', prediction=prediction, image=image_url)

# --- Run App ---
if __name__ == '__main__':
    # For Docker, avoid debug=True in production
    app.run(debug=False, host='0.0.0.0', port=5000)
