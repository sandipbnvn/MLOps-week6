# Iris Classifier API

A machine learning API service that classifies Iris flowers based on their sepal and petal measurements using a pre-trained scikit-learn model. This project demonstrates MLOps practices with Docker containerization and Kubernetes deployment configurations.

## 🌸 Project Overview

This API provides a RESTful interface for Iris flower classification using the classic Iris dataset. The service is built with FastAPI and includes:

- **ML Model**: Pre-trained scikit-learn classifier for Iris species prediction
- **API Endpoints**: Health checks, root endpoint, and prediction endpoint
- **Containerization**: Docker support for easy deployment
- **Kubernetes**: Production and staging deployment configurations
- **Testing**: Comprehensive test suite for API validation

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker (optional, for containerized deployment)
- Kubernetes cluster (optional, for K8s deployment)

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MLOps-week6
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the API locally**
   ```bash
   uvicorn iris_fastapi:app --host 0.0.0.0 --port 8200 --reload
   ```

4. **Test the API**
   ```bash
   python test.py
   ```

The API will be available at `http://localhost:8200`

## 📚 API Documentation

### Endpoints

#### GET `/`
Returns a welcome message.
```json
{
  "message": "Welcome to the Iris Classifier API!v4"
}
```

#### GET `/health`
Health check endpoint for monitoring.
```json
{
  "status": "healthy"
}
```

#### POST `/predict/`
Predicts the Iris species based on flower measurements.

**Request Body:**
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Response:**
```json
{
  "predicted_class": "setosa"
}
```

### Interactive API Documentation

Once the API is running, you can access:
- **Swagger UI**: `http://localhost:8200/docs`
- **ReDoc**: `http://localhost:8200/redoc`

## 🐳 Docker Deployment

### Build the Docker Image
```bash
docker build -t iris-api:latest .
```

### Run the Container
```bash
docker run -p 8200:8200 iris-api:latest
```

### Test the Containerized API
```bash
python test.py
```

## ☸️ Kubernetes Deployment

This project includes Kubernetes manifests for both staging and production environments.

### Staging Environment
```bash
kubectl apply -f k8s/staging-deployment.yaml
```

### Production Environment
```bash
kubectl apply -f k8s/production-deployment.yaml
```

### Deployment Features

- **Resource Limits**: Configured CPU and memory limits
- **Health Checks**: Liveness and readiness probes
- **Load Balancer**: External access via LoadBalancer service
- **Environment Separation**: Different configurations for staging and production

## 🧪 Testing

The project includes a comprehensive test suite (`test.py`) that validates:

- ✅ Service startup and health checks
- ✅ Root endpoint functionality
- ✅ Health endpoint responses
- ✅ Prediction endpoint with sample data

### Running Tests
```bash
# Start the API (in another terminal)
uvicorn iris_fastapi:app --host 0.0.0.0 --port 8200

# Run tests
python test.py
```

## 📁 Project Structure

```
MLOps-week6/
├── artifacts/
│   └── model.joblib          # Pre-trained ML model
├── k8s/
│   ├── production-deployment.yaml
│   └── staging-deployment.yaml
├── iris_fastapi.py           # Main FastAPI application
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker configuration
├── test.py                  # API test suite
└── README.md               # This file
```

## 🔧 Configuration

### Environment Variables
- `PORT`: API port (default: 8200)
- `HOST`: API host (default: 0.0.0.0)

### Model Information
- **Algorithm**: scikit-learn classifier
- **Features**: sepal_length, sepal_width, petal_length, petal_width
- **Classes**: setosa, versicolor, virginica
- **Serialization**: joblib format

## 📊 Monitoring

The API includes built-in monitoring endpoints:
- Health checks for load balancers and monitoring systems
- Standard HTTP status codes for error handling
- Structured JSON responses for easy parsing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is part of the MLOps course materials.

## 🆘 Support

For issues and questions:
1. Check the API documentation at `/docs`
2. Review the test suite for usage examples
3. Open an issue in the repository

---

**Version**: v4  
**Last Updated**: 2024 