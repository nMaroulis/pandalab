from sklearn.svm import SVR
from library.modelling.models.base_model import BaseModel


class SVRModel(BaseModel):

    def __init__(self, params=None, training_params=None):
        # params = {"kernel": params[0], "C": params[1], "epsilon": params[2], "degree": params[3], "tol": params[4]}
        super().__init__("SVR", SVR(**params), params, training_params)
