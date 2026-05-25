import os
import sys
import yaml
import boto3
import logging  # 🌟 Added built-in logging module
import sagemaker
from src.pipeline.pipeline import get_sagemaker_pipeline_config
from src.exception import CustomException

# Configure logging to display outputs on both the console and a physical log file
logging.basicConfig(
    level=logging.INFO,
    format="[ %(asctime)s ] %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout), # 🌟 Enables live terminal log tracking
        logging.FileHandler(os.path.join(os.getcwd(), "logs", "production_deploy.log"))
    ]
)

def execute_production_pipeline():
    logging.info("## [START] AWS SageMaker Cloud Pipeline deployment and execution initialized.")
    
    try:
        # 1. Parse configuration parameters from the config.yaml file
        config_path = os.path.join(os.getcwd(), "config", "config.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
        region = config["aws_infrastructure"]["region"]
        bucket_name = config["aws_infrastructure"]["s3_bucket_name"]

        logging.info(f"AWS Region: {region} | S3 Bucket: {bucket_name}")

        # Explicit Boto3 + SageMaker Session Setup
        boto_session = boto3.Session(region_name=region)
        sagemaker_session = sagemaker.Session(boto_session=boto_session)
        
        # Enterprise execution role ARN fallback configured for local testing context
        role = "arn:aws:iam::517728568420:role/service-role/AmazonSageMaker-ExecutionRole"
        
        # 2. Extract and load the structural pipeline configuration DAG graph
        logging.info("Loading core pipeline workflow configurations...")
        pipeline = get_sagemaker_pipeline_config()
        
        # 3. Compile and upload the pipeline blueprint directly to AWS SageMaker (Upsert)
        logging.info("Uploading pipeline blueprint architecture straight to AWS SageMaker engine...")
        pipeline.upsert(
            role_arn=role,
            description="Enterprise-grade automated end-to-end pipeline predicting student math scores."
        )
        
        # 4. Fire the remote execution trigger to start the cloud orchestration job (Trigger Run)
        logging.info("Triggering remote pipeline execution on AWS Cloud infrastructure...")
        execution = pipeline.start()
        
        logging.info("## [SUCCESS] AWS SageMaker Cloud Pipeline execution triggered successfully!")
        print(f"\n## [DEPLOYED SUCCESS] Pipeline Execution ARN: \n{execution.arn}\n")
        
    except Exception as e:
        logging.error(f"AWS SageMaker pipeline execution failed: {str(e)}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    execute_production_pipeline()
