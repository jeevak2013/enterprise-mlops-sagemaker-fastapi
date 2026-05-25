import os
import sys
import json
import tarfile
import pathlib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from src.logger import logging
from src.exception import CustomException

def run_model_evaluation():
    logging.info("## [START] Evaluation Step for sagemaker model started")
    
    try:
        # 1. paths in SageMaker Processing Job container
        model_path = "/opt/ml/processing/model/model.tar.gz"
        test_path = "/opt/ml/processing/test/test.csv"
        output_dir = "/opt/ml/processing/evaluation"
        
        # 2. extract trained model
        if not os.path.exists(model_path):
            logging.error(f"model folder not found in using local path")
            model_path = os.path.join(os.getcwd(), "artifacts", "model", "xgboost-model.json")
            test_path = os.path.join(os.getcwd(), "artifacts", "test", "test.csv")
            output_dir = os.path.join(os.getcwd(), "artifacts", "evaluation")
            
        logging.info(f"Extracting model from: {model_path}")
        
        
        # load extracted xgboost model
        model = xgb.Booster()
        if model_path.endswith(".tar.gz"):
            with tarfile.open(model_path) as tar:
                  tar.extractall(path=".")
            model.load_model("xgboost-model")
        else:
             model.load_model(model_path)

        logging.info("XGBoost model loaded successfully")
        
        # 3. Read Test Data
        df_test = pd.read_csv(test_path, header=None)
        logging.info(f"Read test data, shape: {df_test.shape}")
        
        # Target (math_score) to be in Column 0
        y_test = df_test.iloc[:, 0].to_numpy()
        X_test = df_test.iloc[:, 1:].to_numpy()
        
        # 4. Execute Predictions
        dtest = xgb.DMatrix(X_test)
        y_pred = model.predict(dtest)
        
        # 5. Metrics Calculation
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        
        logging.info(f"Metrics Calculation -> RMSE: {rmse:.4f} | R2 Score: {r2:.4f}")
        
        # 6. Convert SageMaker Schema into JSON 
        # SageMaker-understand in this format only
        report_dict = {
            "regression_metrics": {
                "rmse": {
                    "value": rmse,
                    "standard_deviation": "NaN"
                },
                "r2_score": {
                    "value": r2,
                    "standard_deviation": "NaN"
                }
            }
        }
        
        # Create outfolder and save JSON
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        evaluation_path = os.path.join(output_dir, "evaluation.json")
        
        with open(evaluation_path, "w") as f:
            f.write(json.dumps(report_dict))
            
        logging.info(f"## [SUCCESS] Evaluatio Report saved successfully: {evaluation_path}")

    except Exception as e:
            logging.error("Error in Evaluation Script")
            raise CustomException(e, sys)

if __name__ == "__main__":
    run_model_evaluation()
