import os, sys
from library.modelling.training.model_handler import get_model_object #, download_model, create_downloadable_model
from sklearn.model_selection import train_test_split
from pandas import read_parquet
import time, json, pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pandas import DataFrame
from database.db_client import update_training_status

# from library.modelling.evaluation.evaluation_output import (show_indications, show_model_scores, show_def_residual_analysis, show_prediction_analysis,)

TRAINING_DIR = 'database/cache_training/'


def get_train_test_sets(input_features_list, selected_target, training_set_size, shuffle_data, data_norm='None'):

    df = read_parquet(TRAINING_DIR + 'training_set.parquet')
    # In case Target was chosen twice
    if selected_target in input_features_list:
        input_features_list.remove(selected_target)
    # Remove Null rows
    all_features = input_features_list + [selected_target]
    model_df = df[all_features]
    old_size = model_df.shape[0]
    model_df = model_df.dropna()
    new_size = model_df.shape[0]
    # Feature Scaling
    if data_norm == 'Standard Scaling':
        scaler_x, scaler_y = StandardScaler(), StandardScaler()
        model_df[input_features_list] = DataFrame(scaler_x.fit_transform(model_df[input_features_list]), columns=input_features_list)
        model_df[[selected_target]] = scaler_y.fit_transform(model_df[[selected_target]])
    elif data_norm == 'MinMax':
        scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
        model_df[input_features_list] = DataFrame(scaler_x.fit_transform(model_df[input_features_list]), columns=input_features_list)
        model_df[[selected_target]] = scaler_y.fit_transform(model_df[[selected_target]])
        # Custom Way
        # scaler_y = [model_df[selected_target].min(), model_df[selected_target].max()]
        # model_df = (model_df - model_df.min()) / (model_df.max() - model_df.min())
    else:
        scaler_y = None

    X = model_df[input_features_list]
    y = model_df[selected_target]
    test_size = round((100 - training_set_size)/100, 2)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=shuffle_data)

    return x_train, x_test, y_train, y_test, old_size, new_size, scaler_y


def training_main(training_set_size=80, shuffle_data=False, input_features_list=None, selected_target=None, data_norm='None', model_choice='XGBoost [Auto]', hyperparams=None):

    try:
        update_training_status('Loading data...', os.getpid())
        # CREATE TRAINING & TEST SET
        x_train, x_test, y_train, y_test, old_size, new_size, scaler = get_train_test_sets(input_features_list, selected_target, training_set_size, shuffle_data, data_norm)

        # SAVE TEST DATA
        x_test.to_parquet(TRAINING_DIR + 'test_x.parquet', compression='gzip')
        y_test.to_frame().to_parquet(TRAINING_DIR + 'test_y.parquet', compression='gzip')

        # SAVE SCALER
        if scaler is not None:
            pkl_file = open(TRAINING_DIR + 'scaler.pkl', 'wb')
            pickle.dump(scaler, pkl_file)
            pkl_file.close()

        update_training_status('Dataset Created', os.getpid())

        print(len(x_train), len(x_test), len(y_train))
        print('✅ Dataset Created')

        # GET MODEL OBJECT
        model_object = get_model_object(model_choice, hyperparams, x_train.shape[1])
        print('✅ Object Created', model_object)

        update_training_status('Model Training in Progress...', os.getpid())

        # TRAIN MODEL
        start = time.time()
        model_object.train(x_train, y_train)
        training_time = time.time() - start
        model_object.training_time = training_time
        print('✅ Model Trained', training_time)

        update_training_status('Model Trained', os.getpid())
        model_object.save_model(TRAINING_DIR)

        # SAVE MODEL OBJECT
        pkl_file = open(TRAINING_DIR + 'model_object.pkl', 'wb')
        pickle.dump(model_object, pkl_file)
        pkl_file.close()

    except Exception as e:
        update_training_status('Training Failed: Error Message - ' + str(e), os.getpid())
        sys.exit()


if __name__ == '__main__':
    print("Training Process Started with PID", os.getpid())

    with open(TRAINING_DIR + "training_params.json", "r") as f:
        training_params_dict = json.loads(f.read())
    f.close()
    print(training_params_dict)
    training_main(training_set_size=training_params_dict['training_set_size'],
                  shuffle_data=training_params_dict['shuffle_data'],
                  input_features_list=training_params_dict['input_features'],
                  selected_target=training_params_dict['target_variable'],
                  data_norm=training_params_dict['data_norm'], model_choice=training_params_dict['model'],
                  hyperparams=training_params_dict['hyperparams'])


    # Read the pickle file
    # picklefile = open('cache_training/model_object.pkl', 'rb')
    # # Unpickle the dataframe
    # model_object = pickle.load(picklefile)
    # # Close file
    # picklefile.close()
    # print(model_object)
    #
    # x_test = [[312, 324, 22, 1], [0.4, 2, 221, 1], [7, 5, 22, 1], [312, 276, 5, 1]]
    # x_test = DataFrame(x_test, columns=['p oil', 'T oil', 'ENG_T_ClntBefPmp', 'ENG_T_ClntAftEng'])
    # print(x_test)
    # y_test = [4, 1, 3, 2]
    # DataFrame({'M engine': y_test})
    #
    # scaler=None
    #
    # y_test, y_pred, mean_absolute_error, rmse, max_error, rae, r2 = model_object.evaluate_model(x_test, y_test, scaler)
    # print(y_test, y_pred, mean_absolute_error, rmse, max_error, rae, r2)
    update_training_status('process exit', os.getpid())
    print("Main Finished")
    sys.exit()
