import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

def prep():
    df = pd.read_csv("Churn_Modelling.csv")
    print("Columns in CSV:", df.columns)
    target_column = "Churn" if "Churn" in df.columns else "target"

    X = pd.get_dummies(df.drop(columns=[target_column]), drop_first=True)
    y = df[target_column]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    selected_features = X[:, :18]  # Select only first 18 features for compatibility
    return train_test_split(selected_features, y, test_size=0.3, random_state=42)

def model(x_train, y_train):
    clf = AdaBoostClassifier(n_estimators=100, random_state=42)
    clf.fit(x_train, y_train)
    return clf

def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    return accuracy_score(y_test, predictions), classification_report(y_test, predictions)

def save_model(model, filename="model.joblib"):
    joblib.dump(model, filename)

def load_model(filename="model.joblib"):
    return joblib.load(filename)

def predict(x_test):
    predictions = load_model().predict(x_test)
    print("Predictions:", predictions)

# Run training pipeline
x_train, x_test, y_train, y_test = prep()
adaboost_model = model(x_train, y_train)
accuracy, report = evaluate_model(adaboost_model, x_test, y_test)
print("Accuracy:", accuracy)
print("Report:\n", report)
save_model(adaboost_model)

