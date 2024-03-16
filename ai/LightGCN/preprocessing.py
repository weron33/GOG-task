import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """
    Method to load data from .tvs file.

    :param path: string with path to desired file.
    :return: pd.Dataframe with data from file.
    """
    return pd.read_csv(path, sep='\t')


def prepare_dataset(users: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """
    Method to prepare dataset to recommend products.

    Step by step:
    1) Merges users with products.
    2) Selects columns and fills not bought products with user_id 0.
    2) Rename columns to model required structure.

    It describes each user by their "dream game" including the following columns:
        - user: User id (int)
        - item: Game id (int)
        - label: Game primal genre (int)
        - time: Total number of minutes spent in game by user (int)

    :param users: pd.DataFrame of user data (must contain columns:
    [user_id, game_id, game_time]).
    :param products: pd.DataFrame of products data (must contain columns:
    [id, genre_1_id]).
    :return: users: pd.DataFrame of user representation by "dream game" parameters.
    """
    data = users.merge(products, left_on='game_id', right_on='id', how='right')
    data = data[['user_id', 'id', 'genre_1_id', 'game_time']].fillna(0)
    data = data.rename(columns={'user_id': 'user', 'id': 'item', 'genre_1_id': 'label', 'game_time': 'time'})
    return data
