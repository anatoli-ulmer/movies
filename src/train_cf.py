# Imports librairies
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import implicit
from implicit.evaluation import train_test_split
# from implicit.evaluation import precision_at_k, mean_average_precision_at_k
from workarounds import precision_at_k, mean_average_precision_at_k



# Import Database
dataset_path = "../data/raw/"
dataset_name = "ml-latest-small"
df_ratings = pd.read_csv(f"{dataset_path}{dataset_name}/ratings.csv")
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
    "factors": 50
}
model = implicit.als.AlternatingLeastSquares(**params)
model.fit(train)


# Evaluate model
# Precision@10
precision = precision_at_k(model, train, test, K=10)
# MAP@10
mapk = mean_average_precision_at_k(model, train, test, K=10)

metrics = {"Precision@10": precision, "MAP@10": mapk}

print(metrics)