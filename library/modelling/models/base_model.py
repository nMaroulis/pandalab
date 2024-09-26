from sklearn.metrics import mean_absolute_error, max_error, mean_squared_error, r2_score
from math import sqrt
from numpy import mean as np_mean, sum as np_sum, abs as np_abs
from library.modelling.training.dataset_handler import reverse_scale_data
from streamlit import warning
from library.modelling.model_exporter.sklearn_onnx import get_onnx_model
from pickle import dump, load
from settings.settings import MODEL_SAVE_TEMP_PATH
from joblib import dump as jb_dump, load as jb_load

def relative_absolute_error(true, pred):
    true_mean = np_mean(true)
    squared_error_num = np_sum(np_abs(true - pred))
    squared_error_den = np_sum(np_abs(true - true_mean))
    rae_loss = squared_error_num / squared_error_den
    return rae_loss


class BaseModel:

    def __init__(self, model_name="model", model=None, params=None, training_params=None):
        self.model_name = model_name
        self.model = model
        self.params = params
        self.training_params = training_params
        self.n_features = 0
        self.training_time = 0.0

    def get_model(self):
        return self.model

    def train(self, x_train, y_train):  # To be Overwritten
        self.n_features = len(x_train.columns)
        self.model.fit(x_train, y_train)
        return

    @staticmethod
    def get_eval_scores(y_test, y_pred, scaler):
        if scaler is not None:
            y_test, y_pred = reverse_scale_data(y_test, y_pred, scaler)

        print(y_test[0:10], y_pred[0:10])
        return (y_test, y_pred, mean_absolute_error(y_test, y_pred), sqrt(mean_squared_error(y_test, y_pred)),
                max_error(y_test, y_pred), relative_absolute_error(y_test, y_pred), r2_score(y_test, y_pred))

    def evaluate_model(self, x_test, y_test, scaler):
        y_pred = self.model.predict(x_test)
        return self.get_eval_scores(y_test, y_pred, scaler)

    def get_model_insights(self):
        warning('Nothing to show.')
        return None

    def export_model_pickle(self):
        try:
            with open(MODEL_SAVE_TEMP_PATH + "ml_model.pkl", "wb") as file:
                dump(self.model, file)  # Dump function is used to write the object into the created file in byte format
            file.close()
            return 1, None
        except Exception as e:
            print('Model Export Exception caught', e)
            return 0, "Model Exportation Failed"

    def export_model_joblib(self):
        try:
            with open(MODEL_SAVE_TEMP_PATH + "ml_model.pkl", "wb") as file:
                jb_dump(self.model, file)  # Dump function is used to write the object into the created file in byte format
            file.close()
            return 1, None
        except Exception as e:
            print('Model Export Exception caught', e)
            return 0, "Model Exportation Failed"

    @staticmethod
    def export_model_keras():
        return 0, 'Keras extension is only supported by DNN Tensorflow models. Choose **.pickle** or **.joblib**.'

    @staticmethod
    def export_model_h5():
        return 0, 'h5 extension is only supported by DNN Tensorflow models. Choose **.pickle** or **.joblib**.'

    def export_model_onnx(self):
        res = get_onnx_model(self.model, self.n_features)
        return res

    def load_model(self, path_dir):
        model_f = open(path_dir + 'model.pkl', 'rb')
        self.model = load(model_f)
        model_f.close()
        return

    def save_model(self, path_dir):
        with open(path_dir + 'model.pkl', "wb") as file:
            dump(self.model, file)
        file.close()

        return
