from streamlit import spinner, session_state
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pandas import DataFrame
from numpy import array as np_array


def get_train_test_sets(input_features_list, selected_target, training_set_size, shuffle_data, data_norm='None'):

    with spinner("Setting up Training/Testing Datasets"):

        # In case Target was chosen twice
        if selected_target in input_features_list:
            input_features_list.remove(selected_target)
        # Remove Null rows
        all_features = input_features_list + [selected_target]
        model_df = session_state.df[all_features]
        old_size = model_df.shape[0]
        model_df = model_df.dropna()
        new_size = model_df.shape[0]
        session_state['input_features_list'] = input_features_list
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


def reverse_scale_data(y, y_pred, scaler):
    if scaler is None:
        pass
    else:
        # Scikit Learn way
        y_tmp = np_array(y).reshape(-1, 1)
        y_tmp = scaler.inverse_transform(y_tmp)
        y = y_tmp.flatten()

        y_tmp = np_array(y_pred).reshape(-1, 1)
        y_tmp = scaler.inverse_transform(y_tmp)
        y_pred = y_tmp.flatten()

        # Custom way
        # print(y)
        # y = (y * (scaler[1] - scaler[0])) + scaler[0]
        # print('After Reverse', y)
        # y_pred = (y_pred * (scaler[1] - scaler[0])) + scaler[0]
    return y, y_pred
