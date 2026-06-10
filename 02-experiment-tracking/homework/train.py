import os
import pickle
import click
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):

    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="train-model"):
        
        mlflow.set_tag("developer", "megh")
        mlflow.log_param("model", "RandomForestRegressor")
        
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        # FIX: Subsample the dataset to prevent Out-Of-Memory crashes in cloud IDEs
        # We take the first 10,000 rows for training, and 5,000 for validation
        X_train_small = X_train[:10000]
        y_train_small = y_train[:10000]
        X_val_small = X_val[:5000]
        y_val_small = y_val[:5000]

        # Removed n_jobs=-1 to keep memory usage low
        rf = RandomForestRegressor(max_depth=10, random_state=0)
        
        # Fit on the smaller dataset
        rf.fit(X_train_small, y_train_small)
        y_pred = rf.predict(X_val_small)

        rmse = root_mean_squared_error(y_val_small, y_pred)
        print(f"RMSE on subset: {rmse}")

if __name__ == '__main__':
    
    mlflow.set_tracking_uri("sqlite:///backend.db")
    mlflow.set_experiment("nyc-taxi-experiment")
    
    run_train()