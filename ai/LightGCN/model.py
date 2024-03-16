import numpy as np
import pandas as pd

from scipy.spatial import distance

from libreco.data import random_split, DatasetPure
from libreco.algorithms import LightGCN
from libreco.evaluation import evaluate

from sklearn.model_selection import train_test_split

from typing import List, Tuple

from ai.LightGCN import preprocessing


class RecommendationModel:
    USERS_DATA_PATH: str = 'data/user_game_time.tsv'
    PRODUCTS_DATA_PATH: str = 'data/product_details.tsv'
    NUM_RECOMMENDATIONS: int = 10

    def __init__(self, metric: str):
        self.metric = metric
        self.dataset, self.data_info = self._preprocess_dataset()
        self.model = LightGCN(
            task="ranking",
            data_info=self.data_info,
            loss_type="bpr",
            embed_size=16,
            n_epochs=3,
            lr=1e-3,
            batch_size=2048,
            num_neg=1,
            device="cpu",
        )

    def _preprocess_dataset(self):
        users = preprocessing.load_data(self.USERS_DATA_PATH)
        products = preprocessing.load_data(self.PRODUCTS_DATA_PATH)
        dataset = preprocessing.prepare_dataset(users, products)
        dataset, data_info = DatasetPure.build_trainset(dataset)
        # No dataset splitting, because it is production ready code.
        return dataset, data_info

    def _preprocess_input(self):
        pass

    def fit(self, *args, **kwargs) -> None:
        self.model.fit(
            self.dataset,
            neg_sampling=True,
            verbose=2,
            metrics=["loss"],
        )

    def recommend_products(self, user_id) -> List[Tuple[int, float]]:
        indices = self.model.recommend_user(user=user_id, n_rec=self.NUM_RECOMMENDATIONS,
                                            filter_consumed=True)
        return [(int(indices[user_id][i]), float(self.model.predict(user_id, indices[user_id][i])[0])) for i in range(len(indices[user_id]))]
