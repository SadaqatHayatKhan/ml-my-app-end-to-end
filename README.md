# Credit Card Fraud Detection ML System

This project implements an end-to-end machine learning system for detecting credit card fraud. The system includes model training, API deployment, and a user-friendly frontend interface.

## Project Structure
```
ml_app_assignment/
├── ml/
│   ├── training.ipynb            # Model training notebook
│   └── model.pkl                 # Saved model
├── backend/
│   └── main.py                   # FastAPI app
├── frontend/
│   └── streamlit_app.py          # Streamlit interface
├── mlflow_tracking/
│   └── tracking_setup.md         # MLflow setup & screenshots
├── requirements.txt
└── README.md
```

## Features
- Machine Learning model trained on Credit Card Fraud Detection dataset
- FastAPI backend for model serving
- Streamlit frontend for user interaction
- MLflow for experiment tracking
- Cloud deployment ready

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the FastAPI backend:
```bash
cd backend
uvicorn main:app --reload
```

4. Run the Streamlit frontend:
```bash
cd frontend
streamlit run streamlit_app.py
```

5. Access MLflow UI:
```bash
mlflow ui
```

## Dataset
The project uses the Credit Card Fraud Detection dataset, which contains transactions made by credit cards. The dataset is highly imbalanced and anonymized, containing only numerical input variables which are the result of a PCA transformation.

## Model Details
- Algorithm: Random Forest Classifier
- Features: 30 principal components (V1-V30), Amount, Time
- Target: Class (1 for fraudulent, 0 for normal)
- Metrics tracked: Accuracy, Precision, Recall, F1-Score, ROC-AUC

## API Documentation
The FastAPI backend provides the following endpoints:
- POST `/predict`: Accepts transaction features and returns fraud probability
- GET `/health`: Health check endpoint

## Frontend
The Streamlit frontend provides an intuitive interface for:
- Input of transaction features
- Real-time prediction results
- Visualization of prediction confidence

## MLflow Tracking
The project uses MLflow to track:
- Model parameters
- Training metrics
- Model artifacts
- Experiment runs

## Deployment
Instructions for deploying to cloud platforms:
1. Azure App Service
2. AWS EC2
3. Docker container (optional)

## Contributing
Feel free to submit issues and enhancement requests. 