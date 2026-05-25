import os
import sys
import yaml
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.logger import logging
from src.exception import CustomException

def run_data_preprocessing():
    logging.info("## [START] preprocessing step for SageMaker Processing Job ##")
    
    try:
        # Reading the config.yaml file
        config_path = os.path.join(os.getcwd(), "config", "config.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
        logging.info("config.yaml read successfully.")  

        # Define input and output paths for SageMaker Processing Job
        input_data_path = "/opt/ml/processing/input/student_data.csv"
        output_train_path = "/opt/ml/processing/train/"
        output_test_path = "/opt/ml/processing/test/"
        
        # verify if the input data path exists, if not, use a local file for testing
        if not os.path.exists(input_data_path):
            logging.warning("SageMaker Processing Job input path not found. Using local file for testing.")
            input_data_path = os.path.join(os.getcwd(), "notebook", "data", "stud.csv")

            logging.warning("SageMaker Processing Job output path not found. Using local file for testing.")
            output_train_path = os.path.join(os.getcwd(), "artifacts", "train")
            output_test_path = os.path.join(os.getcwd(), "artifacts", "test")

        # Data Ingestion: Read the CSV file into a DataFrame
        df = pd.read_csv(input_data_path)
        logging.info(f"Data file read successfully. Shape: {df.shape}")

        # Check for missing values and outliers in the numerical features
        # Your target feature is 'math_score'

        X = df.drop(columns=['math_score'], axis=1)
        y = df['math_score']        

        # Identify numerical and categorical features
        num_features = X.select_dtypes(exclude="object").columns.tolist()
        cat_features = X.select_dtypes(include="object").columns.tolist()
        logging.info(f"Numerical features: {num_features} | Categorical features: {cat_features}")

        
        num_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        cat_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ("num_pipeline", num_transformer, num_features),
            ("cat_pipeline", cat_transformer, cat_features)
        ])

        # Fit and Transform the data
        logging.info("Applying transformations to the data...")

        X_transformed = preprocessor.fit_transform(X)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_transformed, y, test_size=0.2, random_state=42
        )

        # Combine the target variable with the features for both training and testing sets
        train_data = np.hstack((y_train.values.reshape(-1, 1), X_train))
        test_data = np.hstack((y_test.values.reshape(-1, 1), X_test))

        logging.info(f"Training data shape: {train_data.shape}")
        logging.info(f"Testing data shape: {test_data.shape}")

        # Create output directories if they don't exist and save the processed data as CSV files
        os.makedirs(output_train_path, exist_ok=True)
        os.makedirs(output_test_path, exist_ok=True)

        train_file_path = os.path.join(output_train_path, "train.csv")
        test_file_path = os.path.join(output_test_path, "test.csv")

        pd.DataFrame(train_data).to_csv(train_file_path, index=False, header=False)
        pd.DataFrame(test_data).to_csv(test_file_path, index=False, header=False)
        
        logging.info(f"## [SUCCESS] Training and testing data saved successfully: {train_file_path}, {test_file_path} ##")

    except Exception as e:
        logging.error("## [FAILED] Data preprocessing failed. ##")
        raise CustomException(e, sys)

if __name__ == "__main__":
    run_data_preprocessing()
