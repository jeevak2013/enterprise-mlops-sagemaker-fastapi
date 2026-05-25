import os
import sys
from contextlib import asynccontextmanager
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict 
from src.logger import logging
from src.exception import CustomException

# 1. Input Schema Structure updated for Pydantic V2 (Using json_schema_extra)
class StudentDataInput(BaseModel):
    model_config = ConfigDict(
            json_schema_extra={
        "example": {
            "gender": "female",
            "race_ethnicity": "group B",
            "parental_level_of_education": "bachelor's degree",
            "lunch": "standard",
            "test_preparation_course": "none",
            "reading_score": 72,
            "writing_score": 74
            }
        }
    )
    
    gender: str = Field(...)
    race_ethnicity: str = Field(...)
    parental_level_of_education: str = Field(...)
    lunch: str = Field(...)
    test_preparation_course: str = Field(...)
    reading_score: int = Field(..., ge=0, le=100)
    writing_score: int = Field(..., ge=0, le=100)


# 2. Global Model Instance Variable
model_instance = None

# 3. Lifespan Event Handler updated for Modern FastAPI Standards
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_instance
    logging.info("FastAPI server starting up. Loading production model artifact...")
    try:
        model_path = os.path.join(os.getcwd(), "artifacts", "model", "xgboost-model.json")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Production model file not found at: {model_path}")
            
        model_instance = xgb.Booster()
        model_instance.load_model(model_path)
        logging.info("XGBoost production model successfully loaded into memory.")
        yield
    except Exception as e:
        logging.error("An error occurred during the FastAPI startup loading phase.")
        raise CustomException(e, sys)
    finally:
        logging.info("FastAPI server shutting down. Cleaning up allocated memory resources.")

# Initialize the FastAPI Application with Lifespan management
app = FastAPI(
    title="Student Math Score Prediction API",
    description="Enterprise-Grade Model Serving Endpoint using FastAPI & XGBoost",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
def read_root():
    """
    Health check endpoint to verify that the API is fully active and operational.
    """
    return {"status": "healthy", "message": "Student Math Score Serving API is fully operational!"}

@app.post("/predict")
async def predict_math_score(student_input: StudentDataInput):
    """
    Main serving endpoint to process incoming requests and yield model predictions.
    """
    logging.info("New incoming prediction request received.")
    try:
        if model_instance is None:
            raise HTTPException(status_code=503, detail="Prediction model is not ready or initialized!")

        # Using Pydantic V2 standard syntax to parse input fields
        data_dict = student_input.model_dump() 
        logging.info(f"Received input payload: {data_dict}")

        # Initializing a temporary NumPy placeholder array to handle local testing execution
        mock_processed_features = np.zeros((1, 19), dtype=np.float32)
        mock_processed_features[0, 0] = float(data_dict['reading_score'])
        mock_processed_features[0, 1] = float(data_dict['writing_score'])

        # Parsing the feature matrix to DMatrix for XGBoost inference
        dmatrix_input = xgb.DMatrix(mock_processed_features)
        prediction = model_instance.predict(dmatrix_input)
        
        # Formatting and rounding the output prediction metrics
        final_score = float(round(prediction[0], 2))
        logging.info(f"Prediction complete. Predicted student math score: {final_score}")

        return {
            "prediction_status": "success",
            "predicted_math_score": final_score,
            "input_features_received": data_dict
        }
    except Exception as e:
        logging.error("An error occurred inside the FastAPI /predict application route.")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # 🌟 Fix: Pointing to the explicit path module location: 'src.serving.app:app'
    uvicorn.run("src.serving.app:app", host="127.0.0.1", port=8000, reload=True)
