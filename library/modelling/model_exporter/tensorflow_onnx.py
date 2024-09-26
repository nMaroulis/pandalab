# import tensorflow as tf
# from tensorflow.python.compiler.tensorrt import trt_convert as trt
import tf2onnx
from settings.settings import TRAINING_CACHE_DIR
from tensorflow import TensorSpec, float32


def get_onnx_model(model, input_shape):
    try:
        print('Converting Model to ONNX')

        # onnx_str = onnx_model.SerializeToString()
        # with open(MODEL_SAVE_TEMP_PATH + "onnx_model.onnx", "wb") as f:
        #     f.write(onnx_str)
        # f.close()

        input_signature = [TensorSpec(shape=input_shape, dtype=float32)]

        onnx_model, _ = tf2onnx.convert.from_keras(model, opset=11, input_signature=input_signature, output_path=TRAINING_CACHE_DIR + "onnx_model.onnx")
        print('SerializeToString Model to ONNX')

        return True
    except Exception as e:
        print('ONNX ERROR', e)
        return False

# empty iterator fix for onnx for v1.15 of ONNX has to be in place in order not to use 1.14.1
# https://github.com/onnx/tensorflow-onnx/issues/2262


""" 
The code below should be copy and pasted to matlab in order to use the ML Model.
Make sure to have ONN
% Import ONNX Network
net = importONNXNetwork('tensorflow_model.onnx', 'OutputLayerType', 'regression');
% Use the imported model to make predictions
YPred = predict(net, XNew);
"""