import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """
    Method to load data from .tvs file.

    :param path: string with path to desired file.
    :return: pd.Dataframe with data from file.
    """
    return pd.read_csv(path, sep='\t')


def prepare_products(products):
    """
    Method to prepare products DataFrame to recommendation process.

    Step by step:
    1) Splits the "game_modes" column into three other columns based on modes in data.
    2) Converts titles to numbers.
    3) Converts taglines to numbers.
    4) Aggregates products by game id.

    :param products: pd.DataFrame of products data (must contain columns:
    [id, title, series_id, genre_1_id, genre_2_id, genre_3_id, developer_id, publisher_id, price, game_modes, tagline]).
    :return: products: pd.DataFrame with products represented by game parameters.
    """

    def split_modes(modes: str) -> dict:
        """
        Method to split "game_modes" column into three separate columns:
            - Single-player
            - Co-op
            - Multi-player

        Each columns takes values {0; 1} depending on if game allows to play itself in certain mode.

        If data are "NaN" or is not string, method assign 0 value.

        :param modes: string with modes of the game (separated by ",").
        :return: result: dict of modes as keys and values {0; 1} depending on modes game provides.
        """
        result = {'Single-player': 0, 'Co-op': 0, 'Multi-player': 0}
        try:
            if modes.strip():  # Check if modes is not empty or None
                mode_list = modes.split(',')
                for mode in mode_list:
                    if mode.strip() in result:
                        result[mode.strip()] = 1
        except AttributeError:
            # modes is None or not a string
            pass
        return result

    # Splitting modes
    products['game_modes'] = products['game_modes'].apply(lambda x: split_modes(x))
    products = pd.concat([products.drop(columns='game_modes'), products['game_modes'].apply(pd.Series)], axis=1)
    # Converting titles to numbers
    products['title'] = products['title'].astype('category').cat.codes
    # Converting tagline to numbers
    products['tagline'] = products['tagline'].astype('category').cat.codes
    return products


def aggregate_products(products: pd.DataFrame) -> pd.DataFrame:
    """
    Method describes each products by parameters for recommendation. It prepares the following columns:
        - most_freq_series: Most frequent series (int)
        - most_freq_genre_1: Most frequent genre (int)
        - most_freq_genre_2: Second most frequent genre (int)
        - most_freq_genre_3: Third most frequent genre (int)
        - most_freq_dev: Most frequent developer (int)
        - most_freq_pub: Most frequent publisher (int)
        - avg_price: Average game price (float)
        - avg_single_mode: Average single-player mode in game (float)
        - avg_coop_mode: Average co-op mode in game (float)
        - avg_multi_mode: Average multi-player mode in game (float)

    :param products: pd.DataFrame of products data (must contain columns:
    [id, title, series_id, genre_1_id, genre_2_id, genre_3_id, developer_id, publisher_id, price, Single-player, Co-op, Multi-player]).
    :return: products: pd.DataFrame with products represented by game parameters.
    """
    # Aggregate products by game id
    products = products.groupby('id').agg(
        most_freq_series=('series_id', pd.Series.mode),
        most_freq_genre_1=('genre_1_id', pd.Series.mode),
        most_freq_genre_2=('genre_2_id', pd.Series.mode),
        most_freq_genre_3=('genre_3_id', pd.Series.mode),
        most_freq_dev=('developer_id', pd.Series.mode),
        most_freq_pub=('publisher_id', pd.Series.mode),
        avg_price=('price', 'mean'),
        avg_single_mode=('Single-player', 'mean'),
        avg_coop_mode=('Co-op', 'mean'),
        avg_multi_mode=('Multi-player', 'mean')
    )
    return products


def prepare_users(users: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """
    Method to prepare user data to recommend them correct products.

    Step by step:
    1) It merges users with products.
    2) Drops not bought products.
    2) Aggregate users to represent his "dream game".

    It describes each user by their "dream game" including the following columns:
        - most_freq_series: Most frequent series (int)
        - most_freq_genre_1: Most frequent genre (int)
        - most_freq_genre_2: Second most frequent genre (int)
        - most_freq_genre_3: Third most frequent genre (int)
        - most_freq_dev: Most frequent developer (int)
        - most_freq_pub: Most frequent publisher (int)
        - avg_price: Average game price (float)
        - avg_single_mode: Average single-player mode in game (float)
        - avg_coop_mode: Average co-op mode in game (float)
        - avg_multi_mode: Average multi-player mode in game (float)

    :param users: pd.DataFrame of user data (must contain columns:
    [user_id, game_id]).
    :param products: pd.DataFrame of products data (must contain columns:
    [id, series_id, genre_1_id, genre_2_id, genre_3_id, developer_id, publisher_id, price, Single-player, Co-op, Multi-player]).
    :return: users: pd.DataFrame of user representation by "dream game" parameters.
    """
    # Merging users and products
    users = users.merge(products, left_on='game_id', right_on='id', how='right')
    # Dropping not bought products
    users = users.dropna()
    # Aggregating by user_id
    users = users.groupby('user_id').agg(
        most_freq_series=('series_id', pd.Series.mode),
        most_freq_genre_1=('genre_1_id', pd.Series.mode),
        most_freq_genre_2=('genre_2_id', pd.Series.mode),
        most_freq_genre_3=('genre_3_id', pd.Series.mode),
        most_freq_dev=('developer_id', pd.Series.mode),
        most_freq_pub=('publisher_id', pd.Series.mode),
        avg_price=('price', 'mean'),
        avg_single_mode=('Single-player', 'mean'),
        avg_coop_mode=('Co-op', 'mean'),
        avg_multi_mode=('Multi-player', 'mean')
    )
    return users
