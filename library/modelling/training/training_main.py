from streamlit import toast, markdown, caption, session_state, sidebar, expander, columns, write, button, fragment, \
    download_button, dialog, text_input, html, code, warning, error, radio
from library.modelling.evaluation.evaluation_output import (show_indications, show_model_scores,
                                                            show_def_residual_analysis, show_prediction_analysis,)
from library.modelling.training.dataset_handler import get_train_test_sets
from library.modelling.training.model_handler import get_model_object, model_download_dialog
import time, os, pickle, json
from pandas import read_parquet, read_csv
from settings.settings import TRAINING_CACHE_DIR


def check_training_status():
    if os.path.exists(TRAINING_CACHE_DIR + 'model_object.pkl'):
        return True
    else:
        return False


def training_main(training_set_size=80, shuffle_data=False, input_features_list=None, selected_target=None, data_norm='None', model_choice='XGBoost [Auto]', hyperparams=None):


    # sidebar.write('🕒 Model Training in progress')
    # sidebar.markdown('<p style="text-align:center;"><img src="https://i.gifer.com/ZKZg.gif" style="width:48px;height:48px;"</p>', unsafe_allow_html=True)

    # CREATE TRAINING & TEST SET
    # x_train, x_test, y_train, y_test, old_size, new_size, scaler = get_train_test_sets(input_features_list, selected_target, training_set_size, shuffle_data, data_norm)
    # toast('✅ Dataset Created')

    # GET MODEL OBJECT
    # model_object = get_model_object(model_choice, hyperparams, x_train.shape[1])
    # toast('✅ Model Initiated')

    # TRAIN MODEL
    # with spinner(model_choice + " Model Training in progress..."):
    #     start = time.time()
    #     model_object.train(x_train, y_train)
    #     training_time = time.time() - start
    # toast('✅ Model Trained')

    # DOWNLOAD MODEL FROM SIDEBAR
    # download_model(model_object.get_model(), model_choice)

    # PRINT INSTRUCTIONS
    caption(f"""The model's performance in the evaluation phase is a strong indicator of how it will perform in the real world. It's important to understand these metrics to make informed decisions about which model to use for prediction. In the following Evaluation section, a comprehensive evaluation of the model's performance is shown, which will provide a clear understanding of the model's capabilities and limitations.""")
    # LOAD TRAINING PARAMS
    with open(TRAINING_CACHE_DIR + "training_params.json", "r") as f:
        training_params_dict = json.loads(f.read())
    f.close()

    # LOAD MODEL
    picklefile = open(TRAINING_CACHE_DIR + 'model_object.pkl', 'rb')
    model_object = pickle.load(picklefile)
    picklefile.close()

    model_object.load_model(TRAINING_CACHE_DIR)

    # LOAD TEST DATA
    x_test = read_parquet(TRAINING_CACHE_DIR + 'test_x.parquet')
    session_state['input_features_list'] = x_test.columns.tolist()
    y_test = read_parquet(TRAINING_CACHE_DIR + 'test_y.parquet')
    y_test = y_test.values.flatten()

    # LOAD SCALER
    if os.path.exists(TRAINING_CACHE_DIR + 'scaler.pkl'):
        picklefile = open(TRAINING_CACHE_DIR + 'scaler.pkl', 'rb')
        scaler = pickle.load(picklefile)
        picklefile.close()
    else:
        scaler = None

    # MODEL EVALUATION SCORES
    y_test, y_pred, mean_absolute_error, rmse, max_error, rae, r2 = model_object.evaluate_model(x_test, y_test, scaler)
    toast('✅ Model Evaluation was generated')

    # MODEL SCORES
    show_model_scores(mean_absolute_error, rmse, max_error,  rae, r2, model_object.training_time, len(x_test), len(x_test), training_params_dict)
    # y_test_l = y_test  #.to_list()

    # MODEL INDICATIONS
    show_indications(r2, mean_absolute_error, y_test)

    # 2. RESIDUAL PLOTS
    show_def_residual_analysis(y_test, y_pred)

    # 3. PREDICTION ANALYSIS
    show_prediction_analysis(y_test, y_pred)

    # 4. MODEL SPECIFIC STATISTICS
    markdown("<h3 style='text-align: left; color: #48494B;margin-top:2em;'>4. Model Specific Insights</h3>", unsafe_allow_html=True)
    markdown("<h6 style='text-align: center; color: #48494B;'>Different ML Models can provide different Insights for the Training Task or the Data</h6>",
             unsafe_allow_html=True)
    model_object.get_model_insights()

    session_state['ml_model_object'] = model_object

    # DOWNLOAD MODEL
    toast('Model is now ready to Download', icon='✅')
    with sidebar:
        model_download_button()

    return

@fragment
def model_download_button():
    if button('🔗 Download Model'):
        model_download_dialog()
