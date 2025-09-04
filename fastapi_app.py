from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Create FastAPI instance
app = FastAPI()

# Define input data model with 18 features
class PredictionRequest(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float
    feature5: float
    feature6: float
    feature7: float
    feature8: float
    feature9: float
    feature10: float
    feature11: float
    feature12: float
    feature13: float
    feature14: float
    feature15: float
    feature16: float
    feature17: float
    feature18: float

# Load the pre-trained model (ensure the model is in the same directory or specify the path)
try:
    model = joblib.load("model.joblib")  # Make sure to use the correct model format (joblib or pickle)
    print("Model loaded successfully")
except FileNotFoundError:
    print("Error: model.joblib not found.")
    model = None

# Define a POST route for making predictions
@app.post("/predict/")
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not found. Please train the model first.")

    try:
        # Prepare input data for prediction
        input_data = np.array([[
            request.feature1, request.feature2, request.feature3,
            request.feature4, request.feature5, request.feature6,
            request.feature7, request.feature8, request.feature9,
            request.feature10, request.feature11, request.feature12,
            request.feature13, request.feature14, request.feature15,
            request.feature16, request.feature17, request.feature18
        ]])

        # Perform prediction with the model
        prediction = model.predict(input_data)
        
        # Return the result as a JSON response
        return {"prediction": prediction.tolist()}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

