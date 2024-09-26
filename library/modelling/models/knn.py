from sklearn.neighbors import KNeighborsRegressor
from library.modelling.models.base_model import BaseModel


class KNNModel(BaseModel):

    def __init__(self, params=None, training_params=None):
        # params = {"n_neighbors": params.get("n_neighbors"), "weights": params.get("weights"), "algorithm": params.get("algorithm"), "leaf_size": params.get("leaf_size"),
        #           "p": params.get("p"), "metric": params.get("metric")}
        super().__init__("KNN", KNeighborsRegressor(**params), params, training_params)
