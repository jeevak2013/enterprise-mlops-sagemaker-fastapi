# 🚀 Enterprise Student Performance Analytics & MLOps Pipeline

An production-grade, automated end-to-end machine learning orchestration and serving pipeline engineered on AWS SageMaker AI infrastructure and FastAPI.

---

## 🏛️ System Architecture Overview

This repository contains a high-utility, secure, and production-ready MLOps framework designed around the **Principle of Least Privilege (PoLP)** and modular code architecture. It transitions an abstract student dataset into a fully automated cloud-orchestrated training and validation pipeline, featuring localized testing capabilities via a hybrid switching environment.

```text
       [Raw S3 Input]
             │
             ▼
   ┌───────────────────┐
   │   Data Quality    │ ──► (Validates Row Count, Schema Consistency & Range Constraints)
   └─────────┬─────────┘
             │ (Passed)
             ▼
   ┌───────────────────┐
   │   Preprocessing   │ ──► (Sklearn Pipeline: Median Imputer + StandardScaler + OneHotEncoder)
   └─────────┬─────────┘
             │
             ▼
   ┌───────────────────┐
   │  Model Training   │ ──► (XGBoost Estimator optimized on managed AWS Instance)
   └─────────┬─────────┘
             │
             ▼
   ┌───────────────────┐
   │    Evaluation     │ ──► (Calculates R² Score & RMSE Gatekeeper Node)
   └─────────┬─────────┘
             │
             ▼
   ┌──────────────────────────────────────────┐
   │          Quality Gate Condition          │
   ├────────────────────┬─────────────────────┤
   │    If R² >= 0.75   │    If R² < 0.75     │
   │         ▼          │         ▼           │
   │  ┌──────────────┐  │  ┌───────────────┐  │
   │  │Register Model│  │  │   Pipeline    │  │
   │  │  (Registry)  │  │  │   Fail Step   │  │
   │  └──────────────┘  │  └───────────────┘  │
   └────────────────────┴─────────────────────┘
```

---

## 📂 Project Structure Matrix

```text
sagemaker_proj/
├── config/
│   ├── config.yaml          # Central Configuration File (Region, S3 Targets, Multi-Instance Limits)
│   └── iam_policy.json      # Hardened Least Privilege AWS IAM Access Rules (Version 2012-10-17)
├── src/
│   ├── data_quality/
│   │   └── check_rules.py   # Raw Input Schema and Domain Ingestion Verification Gatekeeper
│   ├── pipeline/
│   │   ├── preprocess.py    # Hybrid Cloud/Local Feature Engineering Engine (Outputs Dense NumPy Arrays)
│   │   ├── train.py         # Managed AWS XGBoost distributed training component using Argparse
│   │   ├── evaluate.py      # Standardized Regression Metric Analyzer (Generates evaluation.json)
│   │   ├── bias_check.py    # Localized Fairness Evaluator (Fairlearn implementation with 0.33 MAE Difference)
│   │   └── abalone/
│   │       └── pipeline_run.py # Master Boto3 Execution layer mapping upsert operations to AWS SageMaker
│   └── serving/
│       └── app.py           # Production FastAPI Endpoint with async lifespan caching & full CORS middleware
├── requirements.txt         # Pinned execution packages mapping strict compatibility matrix
└── .vscode/
    └── settings.json        # Workspace specific Python isolated interpreter path mapping overrides
```

---

## 🚀 Installation & Local Environment Setup

### 1. Initialize Virtual Environment
Configure a deterministic sandboxed Python workspace using the host command shell:
```bash
# Create local virtual environment
python -m venv .venv

# Activate environment (Windows Command Prompt)
.venv\Scripts\activate

# Update core dependency resolver tools
python -m pip install --upgrade pip
```

### 2. Install Dependencies
Deploy the strict stable package dependencies version configuration pinning:
```bash
pip install -r requirements.txt
```

### 3. Bind Local Global Environment Path
Ensure the host compiler maps internal package endpoints reliably:
```cmd
set PYTHONPATH=src
```

---

## 📊 Local Verification Workflow (Testing the Blocks)

This architecture utilizes a **Hybrid Architecture** allowing developers to debug the entire end-to-end framework locally before provisioning cloud components.

```bash
# 1. Run Input Schema Data Quality Checks
python -m src.data_quality.check_rules

# 2. Run Hybrid Preprocessing Pipeline (Compiles datasets)
python -m src.pipeline.preprocess

# 3. Run XGBoost Model Training Locally (Saves trained binary model)
python -m src.pipeline.train

# 4. Generate Performance Metrics and Validation Threshold Matrices
python -m src.pipeline.evaluate

# 5. Run Fairlearn Demographics & Group Bias Evaluations
python -m src.pipeline.bias_check
```

---

## ⚡ Real-Time Model Serving (FastAPI Layer)

The production-ready real-time inferencing server leverages **Asynchronous Lifespan Caching** ensuring the model binary loads into RAM exactly once at boot time.

### Launch local serving engine:
```bash
python -m src.serving.app
```
* Access active baseline response: `http://localhost:8000/`
* Access interactive Swagger documentation testing suite: **`http://localhost:8000/docs`**

---

## ☁️ Cloud Orchestration Deployment (AWS SageMaker Engine)

### 1. Authenticate with AWS CLI Infrastructure
Ensure your workspace contains active identity profile parameters linked to your target AWS user context:
```cmd
aws configure
```
* **Default Region:** `us-east-1`
* **Output Format:** `json`

### 2. Validate and Compile Workflow Map
Test structural integration parameters against active template structures:
```bash
python -m src.pipeline.pipeline
```

### 3. Upload and Execute Live DAG Pipeline
Deploy and register the workflow blueprint directly into the AWS SageMaker Cloud Engine orchestration layout:
```bash
python -m src.pipeline.abalone.pipeline_run
```
Upon successful execution, the interface prints an active **Global Resource Execution Token (Pipeline ARN)**. Visual progress metrics can be traced under `Amazon SageMaker ➔ Pipelines` via your target AWS Management Console dashboard.

---

## 🔒 Enterprise Security & FinOps Governance

* **Least Privilege Enforcement**: Execution parameters strictly limit IAM policy actions down to specific AWS resources, preventing configuration footprint escapes.
* **Transient Multi-Instance Scoping**: Compute nodes (`ml.t3.medium` and `ml.m5.large`) are decoupled and completely transient—proactively provisioning on-demand and auto-terminating post job execution to minimize infrastructure run costs.
