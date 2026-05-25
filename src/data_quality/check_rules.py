import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException

def verify_data_quality(file_path):
    """
    verify data quality of the input file.
    """
    logging.info(f"Data quality check started for: {file_path}")
    
    try:
        # 1. Check if the file exists

        if not os.path.exists(file_path):
            logging.error(f"Input file not found: {file_path}")
            return False
            
        df = pd.read_csv(file_path)
        
        # 2. Minimum row count check (Row Count Check)  
        min_rows = 100
        if len(df) < min_rows:
            logging.warning(f"Data quality issue: Minimum {min_rows} rows required. Only {len(df)} rows found.")
            return False
            
        # 3. Required columns check (Column Schema Check)
        required_columns = [
            "gender", "race_ethnicity", "parental_level_of_education", 
            "lunch", "test_preparation_course", "reading_score", "writing_score", "math_score"
        ]
        for col in required_columns:
            if col not in df.columns:
                logging.warning(f"Required column missing: '{col}'")
                return False
                
        # 4. Data range check (Data Range Check)
        # Scores should be between 0 and 100
        score_cols = ["reading_score", "writing_score", "math_score"]
        for col in score_cols:
            if not df[col].between(0, 100).all():
                logging.warning(f"Invalid score found in column '{col}': Scores should be between 0 and 100.")
                return False
                
        logging.info("## [SUCCESS] All data quality checks passed!")
        return True

    except Exception as e:
        logging.error("Error occurred while checking data quality.") 
        raise CustomException(e, sys)

if __name__ == "__main__":
    # You can test the function with a sample file path. Make sure to replace 'student_data.csv' with the actual path to your dataset.
    #sample_path = r"D:\mlprojects\sagemaker_proj\notebook\data\stud.csv"
    #verify_data_quality(sample_path)
    pass
