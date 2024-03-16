from ai import models


def create_model(model_name: str, metric):
    """
    Method to create available models by provided name and metric.
    Given dataset will be used to fitting.

    :param model_name: string name of model to be deployed ["knn", "LightGCN"]
    :param metric: string name of metric used to evaluate (avaliable values set as scipy.spatial.distnce, see: https://docs.scipy.org/doc/scipy/reference/spatial.distance.html)
    :return: ready to fine-tuning model with set dataset and metrics
    """
    model = models[model_name]
    return model.RecommendationModel(metric)
