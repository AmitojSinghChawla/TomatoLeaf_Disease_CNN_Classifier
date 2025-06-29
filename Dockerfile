# Use a small official Python image
FROM python:3.9-slim

# Set working directory inside container
WORKDIR /app

# Copy everything from your project to container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask's default port
EXPOSE 5000

# Start the Flask app
CMD ["python", "-m", "app.main"]
