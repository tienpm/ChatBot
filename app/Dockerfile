# 1. Choose a suitable base image
FROM python:3.11-slim  
# 2. Create a working directory within the image
WORKDIR /app

# 3. Copy your project's requirements file
COPY requirements.txt ./requirements.txt

# 4. Install dependencies
RUN pip install -r requirements.txt

# 5. Copy your application code
COPY . .  

# 6. Specify the command to start your FastAPI app (using Uvicorn)
CMD ["python", "app.py"]
