import os
from typing import List, Tuple

import pandas as pd
import os

from src.services import ai_service

# Create model
MODEL_NAME = os.environ.get('MODEL_NAME')
if MODEL_NAME:
    model = ai_service.create_model(MODEL_NAME, 'cosine')
else:
    model = ai_service.create_model('knn', 'cosine')
model.fit()


def get_recommendations(user_id: int) -> List[Tuple[int, float]]:
    if not isinstance(user_id, int):
        raise TypeError(f'Provided user_id is invalid. Value "{user_id}" is not int. [type(user_id)={type(user_id)}]')
    # user_products_id = users.game_id.loc[user_id]
    # if not isinstance(user_products_id, pd.Series):
    #     user_products_id = pd.Series([users.game_id.loc[user_id]])
    # model.fit(user_products_id)
    recommendations = model.recommend_products(user_id)
    return recommendations


if __name__ == '__main__':
    print(get_recommendations(9))
