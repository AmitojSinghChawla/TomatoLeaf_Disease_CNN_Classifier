# Use a small official Python image
FROM python:3.9-slim

# Set working directory inside container
WORKDIR /app

# Copy all project files into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port used by Gunicorn
EXPOSE 5000

# Start the app using Gunicorn with a single worker to reduce memory usage
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "app.main:app"]
