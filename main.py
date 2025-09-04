import argparse
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from model_pipeline import prep, model, evaluate_model
from sklearn.metrics import precision_recall_curve, roc_curve, auc, precision_score, recall_score, f1_score
from elasticsearch import Elasticsearch
import json
import logging
import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CSV_FILE = "Churn_Modelling.csv"
MODEL_FILE = "model.joblib"

# Initialize MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("yasminebahriexperiment")

# Initialize Elasticsearch connection
def init_elasticsearch():
    try:
        # Connect to Elasticsearch Docker container
        es = Elasticsearch(['http://localhost:9200'])
        
        # Print more verbose connection info for debugging
        info = es.info()
        logger.info(f"Connected to Elasticsearch: {info['name']} (version: {info['version']['number']})")
        
        return es
    except Exception as e:
        logger.error(f"Error connecting to Elasticsearch: {str(e)}")
        return None

# Function to send MLflow logs to Elasticsearch
def log_to_elasticsearch(es, run_id, metrics, params, artifacts=None):
    if es is None:
        logger.warning("Elasticsearch connection not available. Skipping log_to_elasticsearch.")
        return
    
    timestamp = datetime.datetime.now().isoformat()
    doc = {
        "timestamp": timestamp,
        "run_id": run_id,
        "metrics": metrics,
        "params": params,
        "artifacts": artifacts or []
    }
    
    try:
        logger.info(f"Sending document to Elasticsearch: {json.dumps(doc, default=str)[:200]}...")
        res = es.index(index="mlflow-logs", document=doc)
        logger.info(f"Log indexed to Elasticsearch: {res['result']} (index: {res['_index']}, id: {res['_id']})")
    except Exception as e:
        logger.error(f"Error indexing to Elasticsearch: {str(e)}")

def read_csv():
    try:
        data = pd.read_csv(CSV_FILE)
        print(f"Data loaded from {CSV_FILE}")
        return data
    except FileNotFoundError:
        print(f"Error: {CSV_FILE} not found.")
        return None

def save_model(clf):
    joblib.dump(clf, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")

def load_model():
    try:
        clf = joblib.load(MODEL_FILE)
        print(f"Model loaded from {MODEL_FILE}")
        return clf
    except FileNotFoundError:
        print(f"Error: {MODEL_FILE} not found. Train the model first.")
        return None

def plot_roc_curve(y_test, y_score):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")  # Save ROC curve as artifact
    plt.show()
    return "roc_curve.png"

def plot_precision_recall_curve(y_test, y_score):
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='b', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig("pr_curve.png")  # Save PR curve as artifact
    plt.show()
    return "pr_curve.png"

def plot_histogram(y_test, y_pred):
    plt.figure(figsize=(10, 8))
    sns.histplot(y_pred, kde=True, color="blue")
    plt.title("Prediction Histogram")
    plt.xlabel("Predicted Values")
    plt.ylabel("Frequency")
    plt.savefig("histogram.png")  # Save histogram as artifact
    plt.show()
    return "histogram.png"

def main():
    parser = argparse.ArgumentParser(description="ML Pipeline")
    parser.add_argument("--prepare", action="store_true", help="Prepare data")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model")
    parser.add_argument("--load", action="store_true", help="Load and evaluate existing model")
    parser.add_argument("--save", action="store_true", help="Save trained model")
    args = parser.parse_args()

    # Initialize Elasticsearch connection
    es = init_elasticsearch()
    if es is None:
        print("Warning: Elasticsearch connection failed. Continuing without Elasticsearch logging.")

    data = read_csv()
    if data is None:
        return

    if args.prepare or args.train or args.evaluate or args.load or args.save:
        try:
            x_train, x_test, y_train, y_test = prep()
            print("Data prepared.")
        except TypeError:
            print("Error: prep() function should not take arguments. Adjust model_pipeline.py.")
            return

    # Start MLflow experiment
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        
        # Dictionary to store metrics for Elasticsearch
        metrics_dict = {}
        params_dict = {}
        artifacts_list = []
        
        if args.train or args.save:
            # Log model training parameters
            params_dict["model_type"] = "AdaBoost"
            params_dict["n_estimators"] = 50
            mlflow.log_param("model_type", params_dict["model_type"])
            mlflow.log_param("n_estimators", params_dict["n_estimators"])
            
            # Train model
            clf = model(x_train, y_train)
            
            # Evaluate model
            accuracy, report = evaluate_model(clf, x_test, y_test)
            y_pred = clf.predict(x_test)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Log metrics to MLflow
            metrics_dict["accuracy"] = accuracy
            metrics_dict["precision"] = precision
            metrics_dict["recall"] = recall
            metrics_dict["f1_score"] = f1
            
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            # Save model
            save_model(clf)
            mlflow.sklearn.log_model(clf, "model", registered_model_name="adaboost_yasmine")
            
            # Generate and log visualizations
            roc_curve_path = plot_roc_curve(y_test, clf.predict_proba(x_test)[:, 1])
            pr_curve_path = plot_precision_recall_curve(y_test, clf.predict_proba(x_test)[:, 1])
            hist_path = plot_histogram(y_test, clf.predict(x_test))
            
            # Log artifacts to MLflow
            mlflow.log_artifact(roc_curve_path)
            mlflow.log_artifact(pr_curve_path)
            mlflow.log_artifact(hist_path)
            
            artifacts_list = [roc_curve_path, pr_curve_path, hist_path]
            
            print("Model trained and evaluated.")

        if args.evaluate:
            clf = model(x_train, y_train)
            accuracy, report = evaluate_model(clf, x_test, y_test)
            print(f"Accuracy: {accuracy}")
            print(f"Classification Report:\n{report}")
            
            y_pred = clf.predict(x_test)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Log metrics to MLflow
            metrics_dict["accuracy"] = accuracy
            metrics_dict["precision"] = precision
            metrics_dict["recall"] = recall
            metrics_dict["f1_score"] = f1
            
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            # Generate and log visualizations
            roc_curve_path = plot_roc_curve(y_test, clf.predict_proba(x_test)[:, 1])
            pr_curve_path = plot_precision_recall_curve(y_test, clf.predict_proba(x_test)[:, 1])
            hist_path = plot_histogram(y_test, clf.predict(x_test))
            
            # Log artifacts to MLflow
            mlflow.log_artifact(roc_curve_path)
            mlflow.log_artifact(pr_curve_path)
            mlflow.log_artifact(hist_path)
            
            artifacts_list = [roc_curve_path, pr_curve_path, hist_path]

        if args.load:
            clf = load_model()
            if clf:
                accuracy, report = evaluate_model(clf, x_test, y_test)
                print(f"Accuracy: {accuracy}")
                print(f"Classification Report:\n{report}")
                
                y_pred = clf.predict(x_test)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                
                # Log metrics to MLflow
                metrics_dict["accuracy"] = accuracy
                metrics_dict["precision"] = precision
                metrics_dict["recall"] = recall
                metrics_dict["f1_score"] = f1
                
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)
                
                # Generate and log visualizations
                roc_curve_path = plot_roc_curve(y_test, clf.predict_proba(x_test)[:, 1])
                pr_curve_path = plot_precision_recall_curve(y_test, clf.predict_proba(x_test)[:, 1])
                hist_path = plot_histogram(y_test, clf.predict(x_test))
                
                # Log artifacts to MLflow
                mlflow.log_artifact(roc_curve_path)
                mlflow.log_artifact(pr_curve_path)
                mlflow.log_artifact(hist_path)
                
                artifacts_list = [roc_curve_path, pr_curve_path, hist_path]
        
        # Send logs to Elasticsearch
        log_to_elasticsearch(es, run_id, metrics_dict, params_dict, artifacts_list)

if __name__ == "__main__":
    main()
