# import libraries
import pandas as pd
import pickle
from pathlib import Path
import argparse
import numpy as np


from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
import mlflow
from mlflow.models.signature import infer_signature

import xgboost as xgb


mlflow.set_tracking_uri("sqlite:///backend.db")
mlflow.set_experiment("green_tripdata_2026-01-experiment")
Path("models/").mkdir(exist_ok=True)

def read_dataframe(color: str, year: int, month: int) -> pd.DataFrame:
    """Reads the parquet file for the given year and month, performs data cleaning and returns a DataFrame.
    
    Args:
        color (str): The color of the trip data (e.g., "green", "yellow").
        year (int): The year of the data to read.
        month (int): The month of the data to read.

    Returns:
        pd.DataFrame: The cleaned DataFrame containing the trip data for the specified year and month
    """
    
    filename = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{color}_tripdata_{year}-{month:02d}.parquet"
    df = pd.read_parquet(filename)
    
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda x: x.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Performs feature engineering on the given DataFrame by creating a new feature 'PU_DO' which is a combination of 'PULocationID' and 'DOLocationID'.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing the trip data.
        
    Returns:
        pd.DataFrame: The DataFrame with the new 'PU_DO' feature added.
    """
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    return df

def create_X(df: pd.DataFrame, dv=None):
    """
    Creates a feature matrix from the given DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing the trip data.
        dv (DictVectorizer, optional): The DictVectorizer to use for feature extraction. If None, a new one will be created.
        
    Returns:
        X (sparse matrix): The feature matrix.
        dv (DictVectorizer): The DictVectorizer used for feature extraction.
    """
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv

def train_model(X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray, dv: DictVectorizer) -> str:
    """Trains an XGBoost regression model using the provided training and validation data, and logs the model and relevant information to MLflow.
    
    Args:
        X_train (pd.DataFrame): The feature matrix for the training data.
        y_train (np.ndarray): The target variable for the training data.
        X_val (pd.DataFrame): The feature matrix for the validation data.
        y_val (np.ndarray): The target variable for the validation data.
        dv (DictVectorizer): The DictVectorizer used for feature extraction.
    
    Returns:
        str: The run ID of the MLflow run.
    """
    with mlflow.start_run():
        
        mlflow.set_tag("Developer Name", "Megh Modi")
        mlflow.log_param("model_type", "XGBoost Regression")
        
        params = {
            "learning_rate": 0.0784901317383746,
            "max_depth": 84,
            "min_child_weight": 0.49036887395182704,
            "objective": "reg:linear",
            "reg_alpha": 0.014718634417135872,
            "reg_lambda": 0.0038376926340167993,
            "seed": 42
        }
        
        mlflow.log_params(params)
        
        booster = xgb.train(
            params=params,
            dtrain=xgb.DMatrix(X_train, label=y_train),
            num_boost_round=100,
            evals=[(xgb.DMatrix(X_val, label=y_val), 'validation')],
            early_stopping_rounds=50
        )
        
        y_pred = booster.predict(xgb.DMatrix(X_val, label=y_val))
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)
        
        with open('models/preprocessor.b', 'wb') as f_out:
            pickle.dump(dv, f_out)
        
        mlflow.log_artifact(local_path="models/preprocessor.b", artifact_path="preprocessor")

        # Create an input example and signature so MLflow records the model signature
        input_example = X_val[:5].toarray() if hasattr(X_val[:5], "toarray") else X_val[:5]
        signature = infer_signature(input_example, y_val[:5])

        mlflow.xgboost.log_model(booster, artifact_path="models", signature=signature, input_example=input_example)
        
        run_id = mlflow.active_run().info.run_id
        return run_id

def run(color: str = "green", year: int = 2026, month: int = 1) -> str:
    """Runs the training pipeline for the specified year and month by reading the data, performing feature engineering, creating the feature matrix, and training the model.
    
    Args:
        color (str, optional): The color of the trip data (e.g., "green", "yellow"). Defaults to "green".
        year (int, optional): The year of the data to use for training. Defaults to 2026.
        month (int, optional): The month of the data to use for training. Defaults to 1.
    Returns:
        str: The run ID of the MLflow run.
    """
    
    color = color.lower()
    
    df_train = read_dataframe(color, year, month)
    
    next_month = month + 1 if month < 12 else 1
    next_year = year if month < 12 else year + 1
    
    df_val = read_dataframe(color, next_year, next_month)

    X_train, dv = create_X(feature_engineering(df_train))
    X_val, _ = create_X(feature_engineering(df_val), dv)

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    run_id = train_model(X_train, y_train, X_val, y_val, dv)
    print(f"Training completed for {color} trip data of {year}-{month:02d}. Model and preprocessor have been logged to MLflow.")
    
    return run_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an XGBoost regression model on the green trip data.")
    parser.add_argument("--color", type=str, default="green", help="The color of the trip data (e.g., 'green', 'yellow').")
    parser.add_argument("--year", type=int, default=2026, required=True, help="The year of the data to use for training (e.g., 2026).")
    parser.add_argument("--month", type=int, default=1, required=True, help="The month of the data to use for training (1-12).")
    args = parser.parse_args()

    run_id = run(color=args.color, year=args.year, month=args.month)
    print(f"MLflow run ID: {run_id}")
    
    with open("run.txt", "w") as f:
        f.write(run_id)