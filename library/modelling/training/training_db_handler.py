import os
from settings.settings import TRAINING_CACHE_DIR


def clear_db():
    if os.path.exists(TRAINING_CACHE_DIR + 'model_object.pkl'):
        os.remove(TRAINING_CACHE_DIR + 'model_object.pkl')
    if os.path.exists(TRAINING_CACHE_DIR + 'test_x.parquet'):
        os.remove(TRAINING_CACHE_DIR + 'test_x.parquet')
    if os.path.exists(TRAINING_CACHE_DIR + 'test_y.parquet'):
        os.remove(TRAINING_CACHE_DIR + 'test_y.parquet')
    if os.path.exists(TRAINING_CACHE_DIR + 'training_params.json'):
        os.remove(TRAINING_CACHE_DIR + 'training_params.json')
    if os.path.exists(TRAINING_CACHE_DIR + 'training_set.parquet'):
        os.remove(TRAINING_CACHE_DIR + 'training_set.parquet')
    if os.path.exists(TRAINING_CACHE_DIR + 'scaler.pkl'):
        os.remove(TRAINING_CACHE_DIR + 'scaler.pkl')
    # models
    if os.path.exists(TRAINING_CACHE_DIR + 'model.pkl'):
        os.remove(TRAINING_CACHE_DIR + 'model.pkl')
    if os.path.exists(TRAINING_CACHE_DIR + 'model.h5'):
        os.remove(TRAINING_CACHE_DIR + 'model.h5')
    if os.path.exists(TRAINING_CACHE_DIR + 'model.keras'):
        os.remove(TRAINING_CACHE_DIR + 'model.keras')
    if os.path.exists(TRAINING_CACHE_DIR + 'model.joblib'):
        os.remove(TRAINING_CACHE_DIR + 'model.joblib')
    if os.path.exists(TRAINING_CACHE_DIR + 'onnx_model.onnx'):
        os.remove(TRAINING_CACHE_DIR + 'onnx_model.onnx')
