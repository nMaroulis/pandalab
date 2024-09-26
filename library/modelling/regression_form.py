from streamlit import (toggle, columns, slider, write, container, button, error, selectbox, session_state,
                       multiselect, divider, radio, code, info, caption, toast, rerun, html)
from database.db_client import create_new_training
from library.modelling.regression_form_helper import hyperparameter_form, get_form_explanation
import json
from subprocess import Popen, PIPE, STDOUT
from library.modelling.training.training_db_handler import clear_db
from settings.settings import TRAINING_CACHE_DIR


def regression_form():

    get_form_explanation()
    divider()

    # 1. DATA
    html("<h4 style='text-align: left; color: #787878;'>1. Training Set Parameters</h4>")
    html("<hr style='text-align: left; width:8em; margin: 0; color: #5a5b5e'></hr>")
    html("<br>")
    html("<h5 style='text-align: left; color: #787878;'>1.1 Training & Test Set</h5>")
    col01, col02 = columns(2)
    with col01:
        training_set_size = slider("Percentage (%) of the DataTable Samples to be Used for Training", 1, 100, 80)
        shuffle_data = toggle('Shuffle Data')
    with col02:
        val_set_size = slider("Percentage (%) of the Train Set to be used for Validation", 0, 100, 0, disabled=True)

    html("<br>")
    html("<h5 style='text-align: left; color: #787878;'>1.2 Define Inputs & Outputs</h5>")
    with container():
        write('Define Model Inputs & Output (Target)')
        col1_in, col2_in = columns([3, 1])
        with col1_in:
            input_features_list = multiselect(label="Define Input Features", options=list(session_state.df._get_numeric_data().columns))
            input_features_all = toggle(label="Use all Features", value=False)
        with col2_in:
            # Choose Target for Training
            target_options = list(session_state.df._get_numeric_data().columns)
            selected_target = selectbox(label='Select Target', options=target_options,
                                           help="The model will be created to provide Feature Importance for the specified Target.")
    html("<br>")
    html("<h5 style='text-align: left; color: #787878;'>1.3 Data Transformation</h5>")
    caption('**MinMax** scales all the data in the range **[0, 1]** where 0 → min and 1 → max. The **Standard Scaling** standardizes features by **removing the mean** and **scaling to unit variance**. \nIt is **highly recommended** that a Transformation is chosen in case of **Neural Network** as the Model of choice.')

    data_norm = radio("Data Scaling", options=['None', 'MinMax', 'Standard Scaling'], horizontal=True,
                      help='**MinMax** scales all the data in the range **[0, 1]** where 0 -> min and 1 -> max.'
                           'The Standard Scaling standardizes features by **removing the mean** and **scaling to unit variance**.')

    # 2. MODEL
    divider()
    html("<h4 style='text-align: left; color: #787878;'>2. Model Selection & Parametrization</h4>")
    html("<hr style='text-align: left; width:8em; margin: 0; color: #5a5b5e'></hr>")
    model_family_choice = radio('Choose Machine Learning Model Type:', options=['Deep Neural Network', 'Machine Learning Model'], horizontal=True)
    if model_family_choice == 'Machine Learning Model':
        model_choice = selectbox('Choose Machine Learning Model', options=['XGBoost [Auto]', 'Multinomial Linear Regression', 'Feed-Forward Neural Network', 'KNN', 'Support Vector Regression (SVR)'])
        if model_choice == "XGBoost [Auto]":
            info(
                "💡 Machine Learning Model choice is set to **Automatic** XGBoost, therefore the system will try to automatically find the best Parametes for the XGBoost Model. The User is also able to change any Hyperparameter manually..")
        hyperparams = hyperparameter_form(model_choice)
    else:  # model_family_choice == Deep Neural Network
        model_choice = selectbox('Choose Deep Neural Network Model', options=['Recurrent Neural Network - LSTM [Auto]', 'Feed-Forward Neural Network [Auto]', 'Convolutional Neural Network [Auto]', 'Custom Architecture'])
        if model_choice != 'Custom Architecture':
            info("💡 Deep Neural Network Model choice is set to **Automatic**, therefore the system will choose the best suited Deep Neural Network Model **Parameters** for the loaded DataTable. The User is also able to change any Hyperparameter manually.")
        hyperparams = hyperparameter_form(model_choice)

    code("🕑 Estimated Execution Time for Dataset with " + str(session_state.df.shape[0]) + " samples - Unknown.", language=None)
    html("<hr style='text-align: left; width:20em; margin: 0; color: #5a5b5e'></hr>")
    cl_submitted = button("Start Training")
    if cl_submitted:
        if input_features_all:
            input_features_list = list(session_state.df._get_numeric_data().columns)
            input_features_list.remove(selected_target)
        elif input_features_all is False and len(input_features_list) <= 0:
            error('**Input Vector is Empty**, Please Indicate the Model Inputs')
            return
        session_state['training_progress'] = 'start'

        # INLINE PARAM
        # training_main(training_set_size, shuffle_data, input_features_list, selected_target, data_norm, model_choice, hyperparams)

        # PROCESS
        clear_db()  # Clear Previous Training Params

        # save dataset
        if selected_target in input_features_list:
            input_features_list.remove(selected_target)
        training_features = input_features_list.copy()
        training_features.append(selected_target)
        session_state.df[training_features].to_parquet(TRAINING_CACHE_DIR + 'training_set.parquet', compression='gzip')
        # save parameters
        training_params = {
            'input_features': input_features_list,
            'target_variable': selected_target,
            'training_set_size': training_set_size,
            'shuffle_data': shuffle_data,
            'data_norm': data_norm,
            'model': model_choice,
            'hyperparams': hyperparams
        }
        with open(TRAINING_CACHE_DIR + "training_params.json", "w") as outfile:
            json.dump(training_params, outfile)
        outfile.close()

        toast('Training Dataset Created', icon="✅")
        toast('Initiating Training Process..', icon="🏁")

        session_state['training_pid'] = Popen(['python', 'library/modelling/training/training_main_process.py'], stdout=PIPE, stderr=STDOUT)
        create_new_training(session_state['training_pid'].pid)

        print('VS :: Regression Form :: Spawned training with PID', session_state['training_pid'])  # os.getcwd()
        # print('Communicate', session_state['training_pid'].communicate())

        session_state['training_in_progress'] = True
        rerun()
