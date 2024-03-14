# Senior Data Engineer (GOG.com)

## Task

You have been approached by a Product Manager who wants to add a new section on the store's front page. They want the section to contain 10 products recommended to the currently logged in user. The products should be shown from the best to the worst fit. While you're not in charge of displaying the recommendations on the website, the frontend devs will need rating for each recommended product so that they can order them properly. The datasets that are available to you are the game details, and user game interactions. More information on the datasets can be found in the [Datasets](#datasets) section below.

Don't worry too much about the performance metrics achieved by your final model. Your solution and approach to solving this problem is more important to us than the final model or its metrics.

## Step by step

1. Select performance metrics used for the model evaluation.
2. Select model, or models, for recommending products to the users.
3. Perform hyperparameter search for the selected performance metrics and models.
4. Train the final model using the best hyperparameters and the provided datasets.
5. Calculate selected performance metrics for the final trained model.
6. Prepare a function that takes in the user ID and returns a list of 10 recommended games and their ratings. For more details see [Function template](#function-template) section.

## Guidelines

### Datasets

#### game_details.tsv

A tab-separated value file containing 1 000 random products from GOG catalog and 11 columns.

- **id** - Integer used for unique game identification.
- **title** - Human readable game title.
- **series_id** - Integer representing a game series the game belongs to.
- **genre_1_id** - Primary genre of the game.
- **genre_2_id** - Secondary genre of the game.
- **genre_3_id** - Tertiary genre of the game.
- **price** - Game price in USD cents.
- **developer_id** - Integer representing the game's developer.
- **publisher_id** - Integer representing the game's publisher.
- **game_modes** - Comma separated list of supported game modes (Single-player,Multi-player,Co-op).
- **tagline** - Short-form game description.

#### user_game_time.tsv

A tab-separated value file containing a dataset describing user-game interactions in 3 columns.

- **user_id** - Integer used for unique user identification.
- **game_id** - Integer used for unique game identification.
- **game_time** - Integer equal to the time in minutes the user has spent in the game, or `0` if the user owns the game but has never played it.

Notes: 

- This data was generated using random generation from normal, uniform and exponential distribiutions. We put effort into generating data that resemble organic user interactions, but expect it to be biased and favor some models over others. Remember however, your solution and approach to solving this problem is more important to us than the final solution.
- **game_id** from this dataset maps to **id** in the *game_details.tsv*.

### Function template

The function for generating recommendations should have the following signature.

```python
from typing import List, Tuple

def get_recommendations(user_id: int) -> List[Tuple[int, float]]:
    ...
```

### Submission

- Our preferred method of solution submission is a link to a Git repository.
- If you don't want to share a Git repository with us, you can send us a zip archive of the project.
- If none of the above options suit you, you can send us a Jupyter Notebook with the solution.
