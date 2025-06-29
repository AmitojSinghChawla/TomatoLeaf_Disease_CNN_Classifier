
# 🍅 Tomato Leaf Disease Classifier
A deep learning web application to classify tomato leaf diseases from uploaded images using a ResNet18 CNN model. Deployed via Docker on Render.

## 🚀 Live Demo
🔗 [Try it Live]([https://tomatoleaf-disease-cnn-classifier.onrender.com])

Note: First load may take 30–60 seconds due to Render’s free-tier spin-up delay.

## 🧠 Project Overview
This project leverages transfer learning using a fine-tuned ResNet18 model trained on a dataset of 10 tomato plant disease categories (including healthy).

The backend is built using Flask, containerized with Docker, and deployed using Render.

## 🖼️ Supported Disease Classes
1. Tomato__Target_Spot  
2. Tomato__Tomato_mosaic_virus  
3. Tomato__Tomato_YellowLeaf__Curl_Virus  
4. Tomato_Bacterial_spot  
5. Tomato_Early_blight  
6. Tomato_healthy  
7. Tomato_Late_blight  
8. Tomato_Leaf_Mold  
9. Tomato_Septoria_leaf_spot  
10. Tomato_Spider_mites_Two_spotted_spider_mite

## 📂 Project Structure

```bash
├── app/
│   ├── main.py                # Flask application
│   ├── utils.py               # Model loading and prediction logic
│   ├── best_resnet_model.pth  # Saved model checkpoint
│   ├── templates/
│   │   └── index.html         # UI template (Jinja2)
│   └── static/
│       ├── style.css          # Optional custom styles
│       └── uploads/           # Stores uploaded images
├── Dockerfile                 # Docker configuration
├── requirements.txt           # Python dependencies
├── .dockerignore
└── README.md
```


## ⚙️ Tech Stack
* Python 3.9
* PyTorch & Torchvision
* Flask
* Docker
* Render (Free Web Service Hosting)

## 🧪 Model Details
* Architecture: ResNet18
* Training type: Transfer Learning
* Loss: CrossEntropyLoss
* Optimizer: Adam
* Input size: 256x256 RGB

## 🐳 Running Locally with Docker
1. Clone the repository
   ```bash
   git clone https://github.com/your-username/tomato-disease-classifier.git
   cd tomato-disease-classifier
   ```

2. Build Docker image
   ```bash
   docker build -t tomato-app .
   ```

3. Run the container
   ```bash
   docker run -p 5000:5000 tomato-app
   ```

   Then open: [http://localhost:5000](http://localhost:5000)

## 📦 Deployment (Render)
This app is Dockerized and deployed on Render as a Web Service:

* Deployment Type: Web Service
* Port exposed: 5000
* Persistent storage: ❌ Not required
* Auto-redeploys on every git push to the connected GitHub repo
* 🔗 [https://tomatoleaf-disease-cnn-classifier.onrender.com](https://tomatoleaf-disease-cnn-classifier.onrender.com)

⚠️ First load may take up to 60 seconds due to Render’s free-tier spin-up delay.

## 📌 Notes
* The model expects RGB images, resized to 256x256.
* The static/uploads/ folder is cleared before every new prediction.
* Project was developed and tested on Windows 11 (WSL) with Docker Desktop.

## 🧾 License
This project is open-sourced under the MIT License.

## 👤 Author
Amitoj Singh Chawla

B.E. in Data Science & Artificial Intelligence

Passionate about AI, clean code, and efficient deployment pipelines.

udes all the necessary sections and is formatted for clarity and redability. Thank you for your patience, and I apologize for any frustration caused.
