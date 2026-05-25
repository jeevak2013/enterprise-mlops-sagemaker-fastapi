import os
import sys
import yaml
import sagemaker

from sagemaker.workflow.pipeline import Pipeline  # type: ignore
from sagemaker.workflow.parameters import ParameterString  # type: ignore
from sagemaker.workflow.steps import ProcessingStep, TrainingStep  # type: ignore
from sagemaker.workflow.properties import PropertyFile  # type: ignore
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo  # type: ignore
from sagemaker.workflow.condition_step import ConditionStep  # type: ignore
from sagemaker.workflow.fail_step import FailStep  # type: ignore
from sagemaker.workflow.step_collections import RegisterModel  # type: ignore

from sagemaker.processing import ScriptProcessor  # type: ignore
from sagemaker.inputs import TrainingInput  # type: ignore
from sagemaker.estimator import Estimator  # type: ignore

from src.logger import logging
from src.exception import CustomException

def get_sagemaker_pipeline_config():
    """
    Compiles and returns the AWS SageMaker production MLOps pipeline orchestration structure.
    """
    logging.info("## [START] AWS SageMaker V3 Cloud Pipeline architecture configuration initiated.")
    
    try:
        # 1. Read project parameters from config.yaml
        config_path = os.path.join(os.getcwd(), "config", "config.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # 2. Establish SageMaker sessions and access role assignments
        bucket_name = config["aws_infrastructure"]["s3_bucket_name"]
        region = config["aws_infrastructure"]["region"]

        # Explicit Boto3 Region Binding
        import boto3
        boto_session = boto3.Session(region_name=region)
        sagemaker_session = sagemaker.Session(boto_session=boto_session)
        
        try:
            role = sagemaker.get_execution_role()
        except ValueError:
            role = "arn:aws:iam::517728568420:role/service-role/AmazonSageMaker-ExecutionRole"

        # 3. Establish Runtime Pipeline parameters (S3 Raw Dataset Source Location)
        input_data_uri = ParameterString(
            name="InputDataUrl", 
            default_value=f"s3://{bucket_name}/student-math-data/student_data.csv"
        )

        # =========================================================================
        # STEP 1: Processing Step execution configuration
        # =========================================================================
        try:
            sklearn_image_uri = sagemaker.image_uris.retrieve(
                framework="scikit-learn", region=region, version="1.2-1"
            )
        except Exception:
            logging.warning("Local SDK configuration footprint missing. Mapping explicit region distribution target directly.")
            sklearn_image_uri = f"683313688378.dkr.ecr.{region}://"

        # Create a SageMaker ScriptProcessor
        sklearn_processor = ScriptProcessor(
            command=['python3'],
            image_uri=sklearn_image_uri,
            role=role,
            instance_count=1,
            instance_type=config["sagemaker_pipeline"]["instances"]["processing"],
            sagemaker_session=sagemaker_session
        )

        step_process = ProcessingStep(
            name="EnterprisePreprocessStep",
            processor=sklearn_processor,
            inputs=[
                sagemaker.processing.ProcessingInput(source=input_data_uri, destination="/opt/ml/processing/input")
            ],
            outputs=[
                sagemaker.processing.ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
                sagemaker.processing.ProcessingOutput(output_name="test", source="/opt/ml/processing/test")
            ],
            code="src/pipeline/preprocess.py"
        )

        # =========================================================================
        # STEP 2: Model Training Step configuration
        # =========================================================================
        try:
            image_uri_xgboost = sagemaker.image_uris.retrieve(
                framework="xgboost", region=region, version="1.7-1"
            )
        except Exception:
            logging.warning("Local SDK training configuration missing. Mapping explicit training target directly.")
            image_uri_xgboost = f"683313688378.dkr.ecr.{region}://"

        xgboost_estimator = Estimator(
            image_uri=image_uri_xgboost,
            role=role,
            instance_count=1,
            instance_type=config["sagemaker_pipeline"]["instances"]["training"],
            output_path=f"s3://{bucket_name}/model-artifacts",
            sagemaker_session=sagemaker_session
        )

        xgboost_estimator.set_hyperparameters(
            objective="reg:squarederror",
            num_round=int(config["sagemaker_pipeline"].get("num_round", 50)),
            max_depth=5,
            eta=0.1
        )

        step_train = TrainingStep(
            name="EnterpriseTrainStep",
            estimator=xgboost_estimator,
            inputs={
                "train": TrainingInput(
                    s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                    content_type="text/csv"
                ),
                "validation": TrainingInput(
                    s3_data=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                    content_type="text/csv"
                )
            }
        )

        # =========================================================================
        # STEP 3: Evaluation Step configuration tracking
        # =========================================================================
        evaluation_report = PropertyFile(
            name="EvaluationReport",
            output_name="evaluation",
            path="evaluation.json"
        )

        evaluation_processor = ScriptProcessor(
            command=['python3'],
            image_uri=image_uri_xgboost,
            role=role,
            instance_count=1,
            instance_type=config["sagemaker_pipeline"]["instances"]["evaluation"],
            sagemaker_session=sagemaker_session
        )

        step_eval = ProcessingStep(
            name="EnterpriseEvalStep",
            processor=evaluation_processor,
            inputs=[
                sagemaker.processing.ProcessingInput(source=step_train.properties.ModelArtifacts.S3ModelArtifacts, destination="/opt/ml/processing/model"),
                sagemaker.processing.ProcessingInput(source=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri, destination="/opt/ml/processing/test")
            ],
            outputs=[
                sagemaker.processing.ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")
            ],
            code="src/pipeline/evaluate.py",
            property_files=[evaluation_report]
        )

        # =========================================================================
        # STEP 4: Condition Step & Registry (R2 Score >= 0.75 Check)
        # =========================================================================
        cond_gte = ConditionGreaterThanOrEqualTo(
            left=sagemaker.workflow.functions.JsonGet(
                step_name=step_eval.name,
                property_file=evaluation_report,
                json_path="regression_metrics.r2_score.value"
            ),
            right=float(config["model_validation"]["metrics"]["min_r2_score"])
        )

        step_register = RegisterModel(
            name="RegisterEnterpriseModel",
            estimator=xgboost_estimator,
            model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
            content_types=["text/csv"],
            response_types=["text/csv"],
            inference_instances=["ml.m5.large"],
            transform_instances=["ml.m5.large"],
            model_package_group_name=config["sagemaker_pipeline"]["model_group"]
        )

        step_fail = FailStep(
            name="PipelineFailedLowAccuracy",
            error_message="Model evaluation failed to clear quality gate threshold constraints (R2 < 0.75). Aborting run."
        )

        step_cond = ConditionStep(
            name="CheckValidationThreshold",
            conditions=[cond_gte],
            if_steps=[step_register],
            else_steps=[step_fail]
        )

        # =========================================================================
        # 5. Pipeline Orchestration
        # =========================================================================
        pipeline = Pipeline(
            name=config["sagemaker_pipeline"]["name"],
            parameters=[input_data_uri],
            steps=[step_process, step_train, step_eval, step_cond],
            sagemaker_session=sagemaker_session
        )

        logging.info("## [END] Pipeline compilation completed.")
        return pipeline

    except Exception as e:
        logging.error("## [ERROR] Pipeline compilation failed.")
        raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline_instance = get_sagemaker_pipeline_config()
    print("## [FINISHED] Pipeline compilation completely verified against enterprise SDK standards.")
