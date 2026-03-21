# API Documentation

## REST API

The project provides a FastAPI-based REST API for real-time text classification.

### Starting the Server

```bash
# Set environment variables (optional)
export MODEL_PATH="models/transformer"
export MODEL_TYPE="transformer"
export PORT=8000
export HOST="0.0.0.0"

# Start the server
python scripts/serve_api.py
```

### Endpoints

#### GET `/`
Returns API information and model status.

**Response:**
```json
{
  "message": "Multilingual AI-Human Text Detection API",
  "model_loaded": true,
  "model_type": "transformer"
}
```

#### GET `/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### POST `/predict`
Classify a single text.

**Request:**
```json
{
  "text": "Your text to classify here",
  "return_probabilities": false
}
```

**Response:**
```json
{
  "text": "Your text to classify here",
  "prediction": "Human",
  "confidence": 0.89,
  "probabilities": {
    "Human": 0.89,
    "AI": 0.11
  }
}
```

#### POST `/predict_batch`
Classify multiple texts.

**Request:**
```json
{
  "texts": [
    "First text to classify",
    "Second text to classify"
  ],
  "return_probabilities": true
}
```

**Response:**
```json
{
  "predictions": [
    {
      "text": "First text to classify",
      "prediction": "Human",
      "confidence": 0.92,
      "probabilities": {
        "Human": 0.92,
        "AI": 0.08
      }
    },
    {
      "text": "Second text to classify",
      "prediction": "AI",
      "confidence": 0.87,
      "probabilities": {
        "Human": 0.13,
        "AI": 0.87
      }
    }
  ]
}
```

### Error Responses

#### 503 Service Unavailable
```json
{
  "detail": "Model not loaded"
}
```

#### 400 Bad Request
```json
{
  "detail": "Batch size too large (max 100)"
}
```

#### 500 Internal Server Error
```json
{
  "detail": "Prediction error: [error message]"
}
```

## Python API

### Data Loading

```python
from multilingual_ai_detection.data import load_multilingual_dataset

# Load the dataset
dataset = load_multilingual_dataset("data", seed=42)
```

### Model Loading and Inference

```python
from multilingual_ai_detection.models import load_model

# Load a trained model
model = load_model("models/transformer", "transformer")

# Single prediction
predictions = model.predict(["Your text here"])
probabilities = model.predict_proba(["Your text here"])

print(f"Prediction: {'Human' if predictions[0] == 0 else 'AI'}")
print(f"Confidence: {max(probabilities[0]):.3f}")
```

### Training

```python
from multilingual_ai_detection.training import train_transformer_model
from multilingual_ai_detection.data import load_multilingual_dataset

# Load data
dataset = load_multilingual_dataset("data")

# Train model
model = train_transformer_model(
    train_dataset=dataset["train"],
    val_dataset=dataset["validation"],
    model_name="xlm-roberta-base",
    output_dir="models/transformer"
)
```

### Evaluation

```python
from multilingual_ai_detection.evaluation import evaluate_model

# Evaluate model
results = evaluate_model(
    model=model,
    test_data=dataset["test"],
    class_names=["Human", "AI"],
    output_dir="evaluation_results"
)

print(f"F1 Score: {results['overall_metrics']['f1']:.3f}")
```

## Command Line Interface

### Training Scripts

#### Train Statistical Model
```bash
python scripts/train_statistical.py \
    --data-dir data \
    --classifier logistic \
    --max-features 10000 \
    --output-dir models/statistical
```

#### Train Transformer Model
```bash
python scripts/train_transformer.py \
    --data-dir data \
    --model-name xlm-roberta-base \
    --output-dir models/transformer \
    --num-epochs 3 \
    --batch-size 16 \
    --learning-rate 2e-5
```

### Inference Script

```bash
# Single text
python scripts/infer.py \
    --model-path models/transformer \
    --model-type transformer \
    --text "Your text to classify"

# Multiple texts from file
python scripts/infer.py \
    --model-path models/transformer \
    --model-type transformer \
    --text-file input.txt \
    --output-file results.json
```

## Web Demo

### Starting the Demo

```bash
streamlit run scripts/web_demo.py
```

### Features
- Interactive text input
- Real-time classification
- Confidence scores and probabilities
- Support for multiple languages
- Model information and limitations

## Docker Deployment

### Building the Image

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

EXPOSE 8000
CMD ["python", "scripts/serve_api.py"]
```

### Running with Docker

```bash
# Build image
docker build -t ai-detection-api .

# Run container
docker run -p 8000:8000 ai-detection-api
```

## Performance Considerations

### Latency
- Statistical models: < 10ms per prediction
- Transformer models: 50-200ms per prediction
- Batch processing: Reduces latency for multiple texts

### Throughput
- Single GPU: ~100 predictions/second (transformer)
- CPU only: ~20 predictions/second (transformer)
- Statistical: ~1000 predictions/second

### Scaling
- Horizontal scaling with load balancer
- Model caching for multiple workers
- Async processing for high-throughput scenarios

## Security Considerations

### Input Validation
- Text length limits (max 10,000 characters)
- Batch size limits (max 100 texts)
- Sanitization of input text

### Rate Limiting
- Implement rate limiting in production
- Monitor for abuse patterns
- API key authentication for production use

### Model Security
- Validate model outputs
- Monitor for adversarial inputs
- Regular model updates for evolving AI patterns