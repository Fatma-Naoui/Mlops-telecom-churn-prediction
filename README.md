# Telecom Churn Prediction — MLOps Pipeline

This project implements an **end-to-end MLOps pipeline** for predicting telecom customer churn. The pipeline ensures scalability, reproducibility, and reliability by covering the full machine learning lifecycle: **data preparation, model training, deployment, and monitoring**.

---

## Project Overview

The solution operationalizes machine learning with:
- Automated data processing and model training
- Containerized deployment with FastAPI and Streamlit
- Experiment tracking with MLflow
- System and performance monitoring with Elasticsearch & Kibana
- Automation via Makefile and CI/CD pipeline integration
- Notifications (email and system alerts) for training events

---

## Key Features

### **End-to-End Pipeline**
- Data preprocessing, feature engineering, and training  
- Automated workflows for consistency and reproducibility  

### **Model Deployment**
- Predictions served via **FastAPI REST API**  
- Interactive demo with **Streamlit**  
- Containerized using **Docker** and orchestrated with `docker-compose`  

### **Automation**
- **Makefile** for common tasks  
- **Jenkinsfile** for CI/CD integration  

### **Experiment Tracking**
- **MLflow** for logging metrics, parameters, and artifacts  
- Local storage (`mlflow.db`) for experiment history  
- Notifications on model training completion  

### **Monitoring & Logging**
- **Elasticsearch + Kibana** dashboards for system performance  
- Tracking of CPU, memory, latency, and model metrics  
- Alerts for anomalies and performance degradation  

---

## Architecture

1. **Data & Training**  
   - Preprocessing and feature engineering in `model_pipeline.py`  
   - Training managed via `main.py` / `app.py` with MLflow logging  
   - Notifications sent on successful training  

2. **Deployment**  
   - **FastAPI** for serving predictions  
   - **Streamlit** for user-facing visualization  
   - Fully containerized with Docker  

3. **Monitoring**  
   - Metrics and experiments tracked with MLflow  
   - System and logs visualized in Kibana  
   - Alerts for reliability and performance  

---

## Tools & Technologies

- **FastAPI** — REST API for serving predictions  
- **Streamlit** — Web interface for visualization  
- **Docker & docker-compose** — Containerization and orchestration  
- **Makefile** — Task automation  
- **MLflow** — Experiment tracking and model registry  
- **Elasticsearch + Kibana** — Monitoring and log visualization  
- **Notification System** — Email and system alerts  
- **Jenkins** — CI/CD pipeline integration  

---
## Project Structure

```plaintext
├── app.py / main.py             # Entry points for training and execution
├── model_pipeline.py            # Preprocessing & training logic
├── fastapi_app.py               # FastAPI service for model deployment
├── streamlit_app.py             # Streamlit interface for interactive predictions
├── send_email.py / send_notif.py # Notification scripts
├── model.joblib / modelRF.pkl   # Saved models
├── scaler.pkl                   # Preprocessing artifact
├── mlflow.db                    # MLflow experiment tracking database
├── Dockerfile / docker-compose.yml # Containerization setup
├── Jenkinsfile / Makefile       # CI/CD and automation
├── requirements.txt             # Python dependencies
├── data/
│   ├── Churn_Modelling.csv      # Dataset 1
│   └── churn-bigml-20.csv       # Dataset 2
└── README.md                    # Project documentation



