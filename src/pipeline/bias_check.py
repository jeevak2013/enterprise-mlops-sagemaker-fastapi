import os
import sys
import json
import pathlib
import pandas as pd
import numpy as np
import xgboost as xgb
from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.metrics import mean_absolute_error
from src.logger import logging
from src.exception import CustomException

def run_bias_and_fairness_check():
    logging.info("## [START] local model started Bias Check.")
    
    try:
        # 1. create folder path
        model_path = os.path.join(os.getcwd(), "artifacts", "model", "xgboost-model.json")
        # raw test data (To find Gender and Race need raw 'stud.csv' file)
        raw_test_path = os.path.join(os.getcwd(), "notebook", "data", "stud.csv")
        output_dir = os.path.join(os.getcwd(), "artifacts", "bias_report")
        
        if not os.path.exists(model_path) or not os.path.exists(raw_test_path):
            logging.error("மாடல் அல்லது அசல் தரவுக் கோப்பு லோக்கலாகக் கண்டறியப்படவில்லை!")
            sys.exit(1)
            
        # 2. Load model and raw_data
        model = xgb.Booster()
        model.load_model(model_path)
        
        df_raw = pd.read_csv(raw_test_path)
        
        # train test split
        _, df_test = train_test_split_local_replica(df_raw)
        
        # 3. preprocess the input feature and predict
        processed_test_path = os.path.join(os.getcwd(), "artifacts", "test", "test.csv")
        df_processed = pd.read_csv(processed_test_path, header=None)
        
        X_test = df_processed.iloc[:, 1:].to_numpy()
        y_test = df_processed.iloc[:, 0].to_numpy()
        
        dtest = xgb.DMatrix(X_test)
        y_pred = model.predict(dtest)
        
        # 4. Bias Metrics using Fairlearn
        # Based on gender compare mae
        sensitive_feature_gender = df_test['gender']
        
        metric_frame = MetricFrame(
            metrics=mean_absolute_error,
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=sensitive_feature_gender
        )
        
        mae_by_gender = metric_frame.by_group
        mae_difference = metric_frame.difference()
        
        logging.info(f"MAE based on gender :\n{mae_by_gender.to_dict()}")
        logging.info(f"MAE Difference between men and women: {mae_difference:.4f}")
        
        # 5. Save report as json
        bias_report = {
            "fairness_metrics": {
                "sensitive_attribute": "gender",
                "mae_by_group": mae_by_gender.to_dict(),
                "mae_difference_threshold_passed": bool(mae_difference < 2.0), # பிழை வேறுபாடு 2-க்குள் இருக்க வேண்டும்
                "absolute_difference": mae_difference
            }
        }
        
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        report_save_path = os.path.join(output_dir, "bias_report.json")
        
        with open(report_save_path, "w") as f:
            f.write(json.dumps(bias_report, indent=4))
            
        logging.info(f"## [SUCCESS] Bias check report saved: {report_save_path}")

    except Exception as e:
        logging.error("Error happened in bias check.")
        raise CustomException(e, sys)

def train_test_split_local_replica(df):
    # train test split
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    return train, test

if __name__ == "__main__":
    run_bias_and_fairness_check()
