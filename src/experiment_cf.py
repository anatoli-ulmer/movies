# Imports librairies
#from mlflow import MlflowClient
import mlflow
# Imports librairies
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import implicit
from implicit.evaluation import train_test_split
# from implicit.evaluation import precision_at_k, mean_average_precision_at_k
from workarounds import precision_at_k, mean_average_precision_at_k
import pickle
import os

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(FILE_DIR, ".."))

# Set tracking experiment
mlflow.set_tracking_uri("http://127.0.0.1:8080")

# Define experiment name, run name and artifact_path name
experiment = mlflow.set_experiment("cf_implicit")
run_name = "run_1"
artifact_path = "movies_cf_als"


# Import Database
ds_path = os.path.join(PROJECT_DIR, "data/raw/")
ds_name = "ml-latest-small"
df_ratings = pd.read_csv(os.path.join(ds_path, ds_name, "ratings.csv"))
df_ratings


# Transform Data
user_map = {u: i for i, u in enumerate(df_ratings['userId'].unique())}
movie_map = {m: i for i, m in enumerate(df_ratings['movieId'].unique())}
rows = df_ratings['userId'].map(user_map)
cols = df_ratings['movieId'].map(movie_map)
data = df_ratings['rating']
sparse_matrix = coo_matrix((data, (rows, cols)))


# Train model
train, test = train_test_split(sparse_matrix, train_percentage=0.8)
params = {
    "factors": 50,
    "regularization": 0.1,
    "iterations": 15
}
model = implicit.als.AlternatingLeastSquares(**params)
model.fit(train)


# Evaluate model
precision = precision_at_k(model, train, test, K=10) # Precision@10
mapk = mean_average_precision_at_k(model, train, test, K=10) # MAP@10

metrics = {"PrecisionTop10": precision, "MAPTop10": mapk}

print(metrics)


# Store information in tracking server
with mlflow.start_run(run_name=run_name) as run:
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    
    models_path = os.path.join(PROJECT_DIR, "data/models")
    os.makedirs(models_path, exist_ok=True)
    models_file = os.path.join(models_path, "als_model.pkl")
    
    # Save ALS model manually
    with open(models_file, "wb") as f:
        pickle.dump(model, f)

    mlflow.log_artifact(models_file, artifact_path=artifact_path)