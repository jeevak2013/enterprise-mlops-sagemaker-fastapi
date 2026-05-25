import os
import sys
import argparse
import yaml
import pandas as pd
import xgboost as xgb
from src.logger import logging
from src.exception import CustomException

def run_model_training():
    logging.info("## [START] model training step for SageMaker Training Job ##")   
    
    try:
        # 1. Command Line Arguments Parsing for SageMaker Training Job
        # 
        parser = argparse.ArgumentParser()
        
        # Hyperparameters 
        parser.add_argument('--num_round', type=int, default=50)
        parser.add_argument('--max_depth', type=int, default=5)
        parser.add_argument('--eta', type=float, default=0.1)
        
        # SageMaker Default Directories
        parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
        parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
        parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
        parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
        
        args = parser.parse_args()
        logging.info("successfully got the Arguments sent by SageMaker instance.")
        
        if not args.train or not os.path.exists(os.path.join(args.train, "train.csv")):
            logging.warning("Not found SageMaker container training path.So using local path.")
            # data_output folder creation
            args.train = os.path.join(os.getcwd(), "artifacts", "train")
            args.validation = os.path.join(os.getcwd(), "artifacts", "test")

            # local model and ouput path
            if not args.model_dir:
                logging.warning("Not found SM_MODEL_DIR. Using local path")
                args.model_dir = os.path.join(os.getcwd(), "artifacts", "model")
            if not args.output_data_dir:
                logging.warning("Not found SM_OUTPUT_DATA_DIR. Using local path")
                args.output_data_dir = os.path.join(os.getcwd(), "artifacts", "data")
        

        # 2. Load Data Channels
        # csv saved for target as first column read here
        train_file = os.path.join(args.train, "train.csv")
        validation_file = os.path.join(args.validation, "test.csv")
        
        logging.info(f"read training data: {train_file}")
        df_train = pd.read_csv(train_file, header=None)
        df_val = pd.read_csv(validation_file, header=None)
        
        # Target (y) and features (X) split
        y_train = df_train.iloc[:, 0].to_numpy()
        X_train = df_train.iloc[:, 1:].to_numpy()
        
        y_val = df_val.iloc[:, 0].to_numpy()
        X_val = df_val.iloc[:, 1:].to_numpy()

        # 3. for XGBoost transform to DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        watchlist = [(dtrain, 'train'), (dval, 'validation')]

        # 4. set training parameters
        params = {
            'objective': 'reg:squarederror', 
            'max_depth': args.max_depth,
            'eta': args.eta,
            'eval_metric': 'rmse'
        }

        # 5. Train the Model
        logging.info("XGBoost model training started..")
        bst = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=args.num_round,
            evals=watchlist,
            early_stopping_rounds=10
        )
        logging.info("model training successful")

        # 6. Save Model Artifact
        # SageMaker convert model 'SM_MODEL_DIR' to 'model.tar.gz' to upload in s3
        os.makedirs(args.model_dir, exist_ok=True)

        model_output_path = os.path.join(args.model_dir, "xgboost-model.json")
        bst.save_model(model_output_path)
        logging.info(f"## [SUCCESS] model saved successfully: {model_output_path}")

    except Exception as e:
        logging.error("Error occured while training the model")
        raise CustomException(e, sys)

if __name__ == "__main__":
    run_model_training()
