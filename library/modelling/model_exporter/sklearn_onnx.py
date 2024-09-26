import numpy
import onnxruntime as rt
from sklearn.datasets import load_iris, load_diabetes, make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor, DMatrix, train as train_xgb
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn, to_onnx, update_registered_converter
from skl2onnx.common.shape_calculator import (
    calculate_linear_classifier_output_shapes,
    calculate_linear_regressor_output_shapes,
)
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from onnxmltools.convert import convert_xgboost as convert_xgboost_booster
from settings.settings import MODEL_SAVE_TEMP_PATH


# https://onnx.ai/sklearn-onnx/introduction.html


def get_onnx_model(model, input_size):
    try:
        print('Converting Scikit-learn Model to ONNX')

        initial_type = [('float_input', FloatTensorType([None, input_size]))]
        onx = convert_sklearn(model, initial_types=initial_type)
        onnx_str = onx.SerializeToString()
        with open(MODEL_SAVE_TEMP_PATH + "sklearn_model.onnx", "wb") as f:
            f.write(onnx_str)
        f.close()
        return 1
    except Exception as e:
        print('ONNX ERROR', e)
        return 0
