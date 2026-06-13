from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
import pickle

from airflow import DAG
from airflow.operators.python import PythonOperator

import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
import mlflow

# ---------------------------------------------------------
# 1. Core ML Functions (Kept Intact)
# ---------------------------------------------------------

def read_dataframe(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    return df

def create_X(df, dv=None):
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv

# ---------------------------------------------------------
# 2. Airflow Task Logic (Disk I/O added for isolation)
# ---------------------------------------------------------

def extract_data_task(**kwargs):
    """Task 1: Downloads and cleans data, saves as Parquet."""
    execution_date = kwargs['logical_date']
    
    # Hardcoded for testing (Change back to dynamic if needed)
    train_date = datetime(2023, 1, 1)
    val_date = datetime(2023, 2, 1)

    print(f"Pulling Training Data: {train_date.strftime('%Y-%m')}")
    df_train = read_dataframe(year=train_date.year, month=train_date.month)
    
    print(f"Pulling Validation Data: {val_date.strftime('%Y-%m')}")
    df_val = read_dataframe(year=val_date.year, month=val_date.month)

    # Save to disk
    data_folder = Path('data')
    data_folder.mkdir(exist_ok=True)
    
    train_path = str(data_folder / "train_clean.parquet")
    val_path = str(data_folder / "val_clean.parquet")

    df_train.to_parquet(train_path)
    df_val.to_parquet(val_path)

    return {'train_path': train_path, 'val_path': val_path}


def preprocess_data_task(**kwargs):
    """Task 2: Reads Parquet, vectorizes features, saves matrices as Pickles."""
    ti = kwargs['ti']
    paths = ti.xcom_pull(task_ids='extract_data')
    
    # Read previous data
    df_train = pd.read_parquet(paths['train_path'])
    df_val = pd.read_parquet(paths['val_path'])

    # Process features
    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    # Process targets
    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    # Save all matrices and the vectorizer to disk so the next task can use them
    data_folder = Path('data')
    
    file_paths = {
        'X_train': str(data_folder / "X_train.pkl"),
        'X_val': str(data_folder / "X_val.pkl"),
        'y_train': str(data_folder / "y_train.pkl"),
        'y_val': str(data_folder / "y_val.pkl"),
        'dv': str(data_folder / "dv.pkl")
    }

    with open(file_paths['X_train'], 'wb') as f: pickle.dump(X_train, f)
    with open(file_paths['X_val'], 'wb') as f: pickle.dump(X_val, f)
    with open(file_paths['y_train'], 'wb') as f: pickle.dump(y_train, f)
    with open(file_paths['y_val'], 'wb') as f: pickle.dump(y_val, f)
    with open(file_paths['dv'], 'wb') as f: pickle.dump(dv, f)

    return file_paths


def train_model_task(**kwargs):
    """Task 3: Loads matrices, trains XGBoost, logs to MLflow."""
    ti = kwargs['ti']
    paths = ti.xcom_pull(task_ids='preprocess_data')

    # Load everything from disk
    with open(paths['X_train'], 'rb') as f: X_train = pickle.load(f)
    with open(paths['X_val'], 'rb') as f: X_val = pickle.load(f)
    with open(paths['y_train'], 'rb') as f: y_train = pickle.load(f)
    with open(paths['y_val'], 'rb') as f: y_val = pickle.load(f)
    with open(paths['dv'], 'rb') as f: dv = pickle.load(f)

    # Setup MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc-taxi-experiment")

    with mlflow.start_run() as run:
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:linear',
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=30,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        models_folder = Path('models')
        models_folder.mkdir(exist_ok=True)
        
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
            
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        return run.info.run_id

# ---------------------------------------------------------
# 3. DAG Definition
# ---------------------------------------------------------

with DAG(
    dag_id='nyc_taxi_training_pipeline_v3',
    start_date=datetime(2023, 3, 1),
    schedule='@monthly',             
    catchup=False,                   
    tags=['zoomcamp', 'mlops']
) as dag:

    task_1_extract = PythonOperator(
        task_id='extract_data',
        python_callable=extract_data_task,
    )

    task_2_preprocess = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data_task,
    )

    task_3_train = PythonOperator(
        task_id='train_model',
        python_callable=train_model_task,
    )

    # Chain them together sequentially!
    task_1_extract >> task_2_preprocess >> task_3_train