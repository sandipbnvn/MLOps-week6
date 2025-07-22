# MLOps Pipeline for Iris Classification

A production-ready MLOps pipeline for training and evaluating machine learning models on the Iris dataset. This pipeline demonstrates best practices for data processing, model training, evaluation, and automated testing.

## 🚀 Features

- **Modular Architecture**: Clean separation of data processing, training, and evaluation
- **Comprehensive Testing**: Unit tests, integration tests, and data validation
- **Automated CI/CD**: GitHub Actions with CML integration for automated reporting
- **Configuration Management**: YAML-based configuration for easy parameter tuning
- **Logging & Monitoring**: Comprehensive logging throughout the pipeline
- **Model Versioning**: Artifact management and model serialization
- **Performance Tracking**: Accuracy reporting and threshold validation

## 📁 Project Structure

```
mlops-pipeline/
├── src/
│   ├── data/
│   │   ├── data_processor.py      # Data loading and preprocessing
│   │   └── data_validator.py      # Data validation utilities
│   ├── training/
│   │   ├── train_model.py         # Model training pipeline
│   │   └── model_evaluator.py     # Model evaluation and metrics
│   └── utils/
│       ├── config.py              # Configuration management
│       └── logger.py              # Logging utilities
├── tests/
│   ├── test_data.py               # Data validation tests
│   ├── test_model.py              # Model training and evaluation tests
│   └── test_integration.py        # End-to-end pipeline tests
├── config/
│   └── config.yaml                # Configuration file
├── data/
│   └── iris.csv                   # Iris dataset
├── artifacts/                     # Model artifacts (gitignored)
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml        # GitHub Actions workflow
├── requirements.txt               # Python dependencies
├── run_pipeline.py                # Main pipeline runner
└── README.md                      # This file
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd mlops-pipeline
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Quick Start

### Run the Complete Pipeline

```bash
python run_pipeline.py
```

This will:
1. Load and validate the Iris dataset
2. Split data into training and test sets
3. Train a Decision Tree classifier
4. Evaluate the model performance
5. Generate evaluation plots and metrics
6. Save all artifacts

### Run Individual Components

```bash
# Data processing only
python run_pipeline.py --data-only

# Model training only (requires pre-processed data)
python run_pipeline.py --training-only

# Model evaluation only (requires trained model)
python run_pipeline.py --evaluation-only
```

### Run Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
python -m pytest tests/test_data.py -v
```

## ⚙️ Configuration

The pipeline is configured via `config/config.yaml`. Key configuration sections:

### Data Configuration
```yaml
data:
  input_path: "data/iris.csv"
  test_size: 0.4
  random_state: 42
  target_column: "species"
  feature_columns:
    - "sepal_length"
    - "sepal_width"
    - "petal_length"
    - "petal_width"
```

### Model Configuration
```yaml
model:
  type: "DecisionTreeClassifier"
  parameters:
    max_depth: 3
    random_state: 1
  artifact_path: "artifacts/model.joblib"
  metrics_path: "artifacts/metrics.json"
```

### Evaluation Configuration
```yaml
evaluation:
  metrics:
    - "accuracy"
    - "precision_macro"
    - "recall_macro"
    - "f1_macro"
  threshold_accuracy: 0.95
```

## 📊 Model Performance

The pipeline evaluates models using multiple metrics:

- **Accuracy**: Overall classification accuracy
- **Precision (Macro)**: Average precision across all classes
- **Recall (Macro)**: Average recall across all classes
- **F1-Score (Macro)**: Average F1-score across all classes
- **Per-class metrics**: Individual performance for each class
- **Cross-validation**: Robust performance estimation

## 🔄 CI/CD Pipeline

The GitHub Actions workflow automatically:

1. **Runs Tests**: Unit tests and integration tests
2. **Executes Pipeline**: Runs the complete MLOps pipeline
3. **Generates Reports**: Creates comprehensive performance reports using CML
4. **Posts Results**: Comments on PRs with accuracy metrics and visualizations
5. **Uploads Artifacts**: Stores model artifacts and evaluation results

### Workflow Triggers

- **Pull Request**: Validation and testing
- **Manual trigger**: On-demand execution

### Note on Code Quality

The CI/CD pipeline focuses on functional correctness and model performance rather than code style. Code quality checks (flake8, black) have been intentionally removed to prioritize:
- **Functional testing** (unit and integration tests)
- **Pipeline execution** (actual ML pipeline)
- **Performance reporting** (CML integration)

This approach emphasizes results over code formatting, which is often more important in ML projects.

## 📈 CML Integration

The pipeline uses [CML (Continuous Machine Learning)](https://cml.dev/) to:

- Generate beautiful performance reports
- Post results as GitHub comments
- Include confusion matrices and ROC curves
- Track model performance over time

Example CML report:
```markdown
## 📊 Model Performance Report

### 🎯 Accuracy Metrics
**Overall Accuracy:** 0.9833
**Precision (Macro):** 0.9833
**Recall (Macro):** 0.9833
**F1-Score (Macro):** 0.9833
**Meets Threshold:** ✅

### 📈 Per-Class Performance
**Setosa:**
  - Precision: 1.0000
  - Recall: 1.0000
  - F1-Score: 1.0000
```

## 🧪 Testing Strategy

### Unit Tests
- **Data Processing**: Schema validation, data quality checks
- **Model Training**: Model creation, training, prediction
- **Model Evaluation**: Metrics calculation, threshold validation

### Integration Tests
- **End-to-End Pipeline**: Complete workflow validation
- **Artifact Generation**: Model and metrics file verification
- **Error Handling**: Graceful failure handling

### Test Coverage
```bash
# Generate coverage report
python -m pytest tests/ --cov=src --cov-report=html

# View coverage in browser
open htmlcov/index.html
```

## 🔧 Development

### Code Quality

The pipeline includes automated code quality checks:

```bash
# Code formatting
black src/ tests/ run_pipeline.py

# Linting
flake8 src/ tests/ run_pipeline.py --max-line-length=88

# Type checking
mypy src/ --ignore-missing-imports
```

### Adding New Models

To add a new model type:

1. **Update `src/training/train_model.py`**:
   ```python
   elif model_type == 'YourNewModel':
       self.model = YourNewModel(**parameters)
   ```

2. **Update configuration**:
   ```yaml
   model:
     type: "YourNewModel"
     parameters:
       param1: value1
       param2: value2
   ```

3. **Add tests** in `tests/test_model.py`

### Adding New Metrics

To add new evaluation metrics:

1. **Update `src/training/model_evaluator.py`**:
   ```python
   metrics['your_metric'] = your_metric_function(y_true, y_pred)
   ```

2. **Update configuration**:
   ```yaml
   evaluation:
     metrics:
       - "your_metric"
   ```

## 📝 Logging

The pipeline provides comprehensive logging:

- **Console Output**: Real-time progress updates
- **File Logging**: Persistent logs in `artifacts/pipeline.log`
- **Structured Logs**: JSON-formatted for easy parsing

Log levels:
- **INFO**: General progress information
- **WARNING**: Non-critical issues
- **ERROR**: Pipeline failures
- **DEBUG**: Detailed debugging information

## 🚨 Error Handling

The pipeline includes robust error handling:

- **Data Validation**: Comprehensive data quality checks
- **Model Training**: Graceful handling of training failures
- **Evaluation**: Safe metric calculation with fallbacks
- **Artifact Management**: Proper cleanup on failures

## 📦 Artifacts

The pipeline generates several artifacts:

- **`artifacts/model.joblib`**: Trained model with metadata
- **`artifacts/metrics.json`**: Evaluation metrics and results
- **`artifacts/pipeline_results.json`**: Complete pipeline results
- **`artifacts/confusion_matrix.png`**: Confusion matrix visualization
- **`artifacts/roc_curves.png`**: ROC curves (if applicable)
- **`artifacts/pipeline.log`**: Pipeline execution logs

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and add tests
4. **Run tests**: `python -m pytest tests/ -v`
5. **Commit your changes**: `git commit -m 'Add amazing feature'`
6. **Push to the branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Iris Dataset**: R.A. Fisher's classic dataset
- **Scikit-learn**: Machine learning library
- **CML**: Continuous Machine Learning framework
- **GitHub Actions**: CI/CD platform

## 📞 Support

For questions and support:

1. **Check the documentation** in this README
2. **Review the test files** for usage examples
3. **Open an issue** for bugs or feature requests
4. **Create a discussion** for general questions

---

**Happy MLOps! 🚀**