# GOG-Task

### Catalog Structure

This whole repository contains as well experiments and production ready-code. Structure of project is 
```shell
├───ai
│   ├───knn
│   └───LightGCN
├───data
├───docker
├───notebooks
├───src
│   └───services
└───tests
    └───unit

```
- `ai` package contains algorithms that can be used in recommendation system.
- `data` folder carries data files (`tsv` format)
- `docker` catalog contains Dockerfile required to create docker image.
- `notebooks` is an experiments' area. There are some Jupyter notebooks with models.
- `src` is catalog that contains all of necessary code to configure model and make it run.
- `tests` contains a code to test created solution.

### Goal
The goal was to create a recommendation system based on chosen model. Then deploy it with function:
```python
from typing import List, Tuple

def get_recommendations(user_id: int) -> List[Tuple[int, float]]:
    ...
```

In this repository one can find implementation of two algorithms:
1. K - Nearest Neighbours
2. Light GCN

### Getting started
To see methodology of workflow, please checkout notebooks in `notebooks/`.

#### Command Line
The code can be run with raw Python (I was using Python 3.10) with previous standard set up.
```bash
pip install -r requirements
```
Make sure to set up environmental variable `MODEL_NAME`
```bash
#Linux / iOS
export MODEL_NAME=<model_name>
# Windows
set MODEL_NAME=<model_name>
```
Available model names are:
- `knn` 
- `LightGCN`

after that simply run command:
```bash
python -m main.py
```
#### Docker 
In this case one can simply run:
```bash
docker-compose up -d
```
to see logs go with:
```bash
docker logs gog-task
```

Logs present only and output of `get_recommendation` function.
