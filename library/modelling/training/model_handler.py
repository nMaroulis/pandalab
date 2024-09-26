from streamlit import sidebar, spinner, info, error, toast, dialog, write, html, caption, text_input, code, radio, session_state, download_button, warning
from pickle import dumps
from library.modelling.models.knn import KNNModel
from library.modelling.models.svr import SVRModel
from library.modelling.models.linear_regression import LinearRegressionModel
from library.modelling.models.xgboost import XGBoostModel
from library.modelling.models.mlp import MLPModel
from library.modelling.models.dnn.dnn_model import DNNModel
from settings.settings import MODEL_SAVE_TEMP_PATH


def get_model_object(model_choice, hyperparams, input_size):
    if model_choice == "KNN":
        model_object = KNNModel(hyperparams, None)
    elif model_choice == "XGBoost [Auto]":
        model_object = XGBoostModel(hyperparams, None)
    elif model_choice == "Support Vector Regression (SVR)":
        model_object = SVRModel(hyperparams, None)
    elif model_choice == "Feed-Forward Neural Network":
        model_object = MLPModel(hyperparams, None)
    elif model_choice == "Multinomial Linear Regression":
        model_object = LinearRegressionModel(hyperparams, None)
    else:  # DNN
        # model_object = LSTMModel(hyperparams, input_size)
        model_object = DNNModel(hyperparams, input_size)
        model_object.create_model_from_params()

    return model_object


# TODO export model as pickle or joblib or keras or onnx
@dialog("Model Download Manager", width="large")
def model_download_dialog():
    html("<br><h5 style='text-align: left; color: #5a5b5e;padding-bottom:0;'>Download Trained Model</h5>")

    caption('Choose the name of the downloaded file:')
    model_name = text_input('Model Filename', placeholder="Insert Filename...", value='model_filename')

    download_format = radio('Model Type', options=['pickle', 'joblib', 'keras', 'h5', 'ONNX'], horizontal=True, disabled=False)

    write('**Model Export File** Information')
    if download_format == 'pickle':
        caption(
            'This File format is able to be imported in **Python** using the **Pickle Library**, where it will be able to be loaded and start the inferencing.')
        code(
            'from pickle import load\nfile = open("' + model_name + '.pkl")\nmodel=load(file)\nfile.close()\nmodel.predict(TEST_X_INPUTS)',
            language='python', line_numbers=True)

        res, status = session_state['ml_model_object'].export_model_pickle()
    elif download_format == 'joblib':
        caption(
            'This File format is able to be imported in **Python** using the **Joblib Library**, where it will be able to be loaded and start the inferencing.')
        code(
            'from joblib import load\nfile = open("' + model_name + '.pkl")\nmodel=load(file)\nfile.close()\nmodel.predict(TEST_X_INPUTS)',
            language='python', line_numbers=True)
        res, status = session_state['ml_model_object'].export_model_joblib()
    elif download_format == 'keras':
        caption(
            'This File format is able to be imported in **Python** using the **Tensorflow2 Library**, where it will be able to be loaded and start the inferencing.')
        code(
            'from tensorflow import keras as kr\nmodel = kr.load("' + model_name + '.keras")\nmodel.predict(TEST_X_INPUTS)',
            language='python', line_numbers=True)
        res, status = session_state['ml_model_object'].export_model_keras()
    elif download_format == 'h5':
        write(
            'This File format is able to be imported in **Python** using the **Tensorflow2 Library**, where it will be able to be loaded and start the inferencing.')
        caption(
            "Please note that MATLAB requires the **Deep Learning Toolbox** and the **Deep Learning Toolbox Converter for ONNX Model Format support package** add-on for MATLAB. If you don't have these, you can get them from the MathWorks website.")
        code(
            'from tensorflow import keras as kr\nmodel = kr.load("' + model_name + '.h5")\nmodel.predict(TEST_X_INPUTS)',
            language='python', line_numbers=True)
        res, status = session_state['ml_model_object'].export_model_h5()
    else:
        write(
            'Follow the Instruction below on how to integrate and use the downloaded Model to your Matlab environment.')
        write(
            "Please note that MATLAB requires the **Deep Learning Toolbox** and the **Deep Learning Toolbox Converter for ONNX Model Format support package** add-on for MATLAB. If you don't have these, you can get them from the MathWorks website.")
        if len(session_state['ml_model_object'].input_shape_onnx) < 3:
            code(
                f"""model = importONNXNetwork('""" + model_name + f""".onnx',"InputDataFormats", "BC", "OutputDataFormats","BC");  % make sure to have the correct file path\n% x = n*{st.session_state['ml_model_object'].n_features} (samples*features) 2D Matrix with the input Data in the right order\ny_pred = predict(model, x); % y_pred now contains a 1D Vector with the Model's Prediction for each Row of the x Matrix\ny_pred % print predicted values""",
                language='matlab', line_numbers=True)
        else:
            code(
                f"""model = importONNXNetwork('""" + model_name + f""".onnx',"InputDataFormats", "BTC", "OutputDataFormats","BC");  % make sure to have the correct file path\n% x = n*{st.session_state['ml_model_object'].n_timesteps}*{st.session_state['ml_model_object'].n_features} (samples*timesteps*features) 3D Matrix with the input Data in the right order\ny_pred = predict(model, x); % y_pred now contains a 1D Vector with the Model's Prediction for each Row of the x Matrix\ny_pred % print predicted values""",
                language='matlab', line_numbers=True)
        res, status = session_state['ml_model_object'].export_model_onnx()

    ext_dict = {'pickle': '.pkl', 'joblib': '.pkl', 'keras': '.keras', 'h5': '.h5', 'ONNX': '.onnx'}
    download_format = ext_dict[download_format]
    html("<hr style='text-align: left; width:8em; margin: 0; color: #5a5b5e'></hr>")
    if res:
        with open(MODEL_SAVE_TEMP_PATH + "ml_model" + download_format, "rb") as f:
            download_button(
                label="Download Model",
                data=f,
                file_name=model_name + download_format,
                # mime='text/plain',
                type='primary'
            )
    else:
        warning(f"Model cannot be downloaded. {status}", icon="🔗")


# def create_downloadable_model(model_object):
#     with sidebar:
#         with spinner('Creating Downloadable Model'):
#             res = model_object.export_onnx()
#         if res:
#             info('🔗 Model is ready for **Download** in the Settings Tab.')
#             toast('✅ Model is now ready to Download')
#         else:
#             error("🔗 Model cannot be  **Downloaded**.")
#             toast('🚫 Model Export Failed!')
#     return


# def download_model(model, model_choice):
#
#     output_name = model_choice.lower() + "_model.pkl"
#     sidebar.download_button(
#         "🔗 Download Model",
#         data=dumps(model),
#         file_name=output_name,
#     )
#     sidebar.button("Reset Model", type="primary")
#     return
