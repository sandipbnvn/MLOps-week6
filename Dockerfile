# 1. Use official python base image
FROM python:3.10-slim

#2. Set working directory
WORKDIR /app

#3. Copy Files
COPY . /app

#4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

#5. expose port
EXPOSE 8200

#6. Command to run the server
CMD ["uvicorn", "iris_fastapi:app", "--host", "0.0.0.0", "--port", "8200"]