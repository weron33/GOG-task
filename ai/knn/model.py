import pandas as pd

from typing import List, Tuple
from sklearn.neighbors import NearestNeighbors

from ai.knn import preprocessing


class RecommendationModel:
    USERS_DATA_PATH: str = 'data/user_game_time.tsv'
    PRODUCTS_DATA_PATH: str = 'data/product_details.tsv'
    NUM_RECOMMENDATIONS: int = 10

    def __init__(self, metric: str):
        self.metric = metric
        self.model = NearestNeighbors(n_neighbors=self.NUM_RECOMMENDATIONS, algorithm='brute', metric=self.metric)

    def _preprocess_dataset(self) -> pd.DataFrame:
        dataset = preprocessing.load_data(self.PRODUCTS_DATA_PATH)
        dataset = preprocessing.prepare_products(dataset)
        dataset = preprocessing.aggregate_products(dataset)
        return dataset

    def _preprocess_input(self, user_id):
        users = preprocessing.load_data(self.USERS_DATA_PATH)
        products = preprocessing.load_data(self.PRODUCTS_DATA_PATH)
        products = preprocessing.prepare_products(products)
        users_input = preprocessing.prepare_users(users, products)
        return users_input.loc[user_id].values.reshape(1, -1)

    def fit(self, users_products_ids: pd.Series = None) -> None:
        dataset = self._preprocess_dataset()
        if users_products_ids is not None:
            self.model.fit(dataset.loc[~dataset.index.isin(users_products_ids)])
        else:
            self.model.fit(dataset)

    def recommend_products(self, user_id) -> List[Tuple[int, float]]:
        user_input = self._preprocess_input(user_id)
        distances, indices = self.model.kneighbors(user_input)
        distances, indices = distances.reshape(-1), indices.reshape(-1)
        return [(int(indices[i]), float(distances[i])) for i in range(len(indices))]
