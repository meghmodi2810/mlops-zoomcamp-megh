# MLOps Zoomcamp Notes

This repository is a hands-on set of MLOps examples built around the common lifecycle of a machine learning project: data preparation, experiment tracking, model packaging, deployment, and monitoring.

The goal is not just to train a model, but to make the process reproducible, collaborative, and deployable.

## What this repository covers

- Local model training with notebooks and Python.
- Experiment tracking with MLflow.
- Dataset and model versioning with DVC.
- Artifact storage with Amazon S3 or local filesystem.
- Containerized execution with Docker.
- CI/CD automation for testing, building, and deployment.
- A practical workflow for moving from notebook experiments to a production service.

## Conventional MLOps workflow

The standard end-to-end flow looks like this:

1. Collect and validate data.
2. Version the dataset with DVC.
3. Train models in a reproducible environment.
4. Track experiments, metrics, parameters, and artifacts with MLflow.
5. Register the best model.
6. Package the model or service in Docker.
7. Deploy to a target environment such as AWS.
8. Monitor performance and retrain when the data or metrics drift.

## Tooling in the stack

### DVC

DVC is used to track data files and intermediate artifacts without storing them directly in Git. This makes large datasets reproducible and easy to share across the team.

Typical responsibilities:

- Track raw and processed datasets.
- Reproduce pipelines from source data to model output.
- Store large files in remote storage such as S3.

### MLflow

MLflow is used to track experiments and organize model training runs.

Typical responsibilities:

- Log parameters, metrics, tags, and artifacts.
- Save model signatures and input examples.
- Compare runs across experiments.
- Register and promote models.

For collaborative work, the MLflow tracking server can use:

- Backend store: PostgreSQL, MySQL, or SQLite for local development.
- Artifact store: Amazon S3 for shared model files and run artifacts.

### Docker

Docker makes training jobs, batch jobs, and model services portable.

Typical responsibilities:

- Freeze the Python environment.
- Run the same code locally and in CI.
- Package inference services for deployment.

### AWS S3

S3 is a common remote store for both DVC and MLflow artifacts.

Typical responsibilities:

- Store datasets managed by DVC.
- Store MLflow model artifacts.
- Share outputs across the team and deployment environments.

### CI/CD

CI/CD automates checks and releases.

Typical responsibilities:

- Run tests and linting on every change.
- Build Docker images.
- Push images to a registry.
- Deploy the service after validation.

## Recommended project flow

If you want a clean conventional MLOps setup, use this sequence:

### 1. Data versioning

Store the dataset in a raw folder and track it with DVC.

Example:

```bash
dvc add data/raw.csv
git add data/raw.csv.dvc .gitignore
git commit -m "Track raw data with DVC"
```

### 2. Training and experiment tracking

Train the model from a script or notebook and log every run with MLflow.

Log at least:

- parameters
- metrics
- model artifact
- model signature
- input example
- custom tags such as team, dataset version, or experiment name

Example MLflow pattern:

```python
with mlflow.start_run():
	mlflow.log_params(params)
	mlflow.log_metrics(metrics)
	mlflow.sklearn.log_model(
		model,
		artifact_path="model",
		signature=signature,
		input_example=input_example,
	)
```

### 3. Model packaging

Wrap the trained model in a service, usually with Flask, FastAPI, or a batch scoring script.

Dockerize the app so the exact runtime can be reused across local development, CI, and deployment.

### 4. Deployment

Deploy the container to a cloud runtime or VM.

Common options:

- AWS EC2
- ECS or EKS
- Elastic Beanstalk
- A simple container host for demo purposes

### 5. Monitoring and iteration

Track production metrics, latency, failures, and data drift. When the system degrades, update the data pipeline and retrain.

## Example architecture

```text
Raw data -> DVC -> Training pipeline -> MLflow tracking -> Model registry
											  |                      |
											  v                      v
										  Docker image           S3 artifacts
											  |
											  v
										Deployment target
											  |
											  v
									   Monitoring / retraining
```

## Repository layout

This repo currently contains notebook-based examples for MLflow and introductory MLOps workflows.

- `01-intro/` for basic intro material.
- `02-experiment-tracking/` for MLflow examples.
- `data/` for local datasets used in examples.
- `requirements.txt` for Python dependencies.

## Local setup

Create a virtual environment and install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Launch Jupyter to work with the notebooks:

```bash
jupyter lab
```

## MLflow example in this repo

The notebook at `02-experiment-tracking/running-mlflow-examples/scenario-1.ipynb` shows how to:

- set an experiment
- log parameters and metrics
- log a model with signature and input example
- log a dataset artifact

That example is a good starting point for building a reproducible training workflow.

## Minimal production checklist

Before calling a model production-ready, make sure you have:

- versioned data
- reproducible training code
- experiment tracking
- model validation
- a serialized model artifact
- a container image
- automated tests
- CI/CD deployment steps
- a monitoring plan

## Next steps

If you want to expand this repo into a full MLOps project, the next useful additions are:

1. a DVC pipeline file for data preparation and training
2. a FastAPI inference service with Docker
3. an MLflow tracking server backed by S3
4. a CI workflow for tests and image builds
5. a deployment target on AWS