import os
from tensorflow.keras.layers import Dense, Flatten, Dropout, LSTM, Conv1D, MaxPooling1D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.regularizers import L1, L2
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredLogarithmicError
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from matplotlib import pyplot as plt
import time
from sklearn.metrics import mean_absolute_error, max_error, mean_squared_error, r2_score
from numpy import mean as np_mean, sum as np_sum, abs as np_abs
from math import sqrt
from library.modelling.training.dataset_handler import reverse_scale_data
from streamlit import pyplot, progress
from pandas import concat
from library.modelling.model_exporter.tensorflow_onnx import get_onnx_model
from settings.settings import MODEL_SAVE_TEMP_PATH
from database.db_client import update_training_status


class DBoutCallback(Callback):

    def __init__(self, all_epochs):
        self.all_epochs = all_epochs
        self.pid = os.getpid()
        print('DNN ', self.pid)

    def on_train_begin(self, logs=None):
        update_training_status('Training Progress: 0%', self.pid)

    def on_train_end(self, logs=None):
        update_training_status('Training Completed', self.pid)

    def on_epoch_end(self, epoch, logs=None):
        update_training_status('Training Progress: ' + str(round(epoch/self.all_epochs, 4)*100) + '%', self.pid)

class DNNModel:

    def __init__(self, params, n_features, name='DNN'):

        self.name = name
        # Model Parameters
        self.n_features = n_features
        self.layers = params.get("layers")
        self.layers_type = params.get("layers_t")
        self.layers_activation_func = params.get("layers_activation_func")
        self.layers_dropout = params.get("layers_dropout")
        self.n_timesteps = params.get("n_timesteps")

        self.l2_reg = params.get("l2_reg")
        self.input_shape = ()
        self.input_shape_onnx = []

        # Training Params
        self.optimizer = params.get("optimizer")
        self.loss_function = self.get_loss_function(params.get("loss_function"))
        self.verbose = 0  # Default
        self.epochs = params.get("num_epochs")  # Default
        self.batch_size = params.get("batch_size")  # Default
        self.patience = params.get("early_stopping")  # Early Stop Default
        self.learning_rate_init = params.get('learning_rate_init')
        # Evaluation Results
        self.history = None
        self.evaluation_score = None
        self.summary = None
        self.training_time = 0.0
        # self.create_model_from_params()  # CREATE MODEL

    @staticmethod
    def get_loss_function(loss):
        if loss == 'Mean Absolute Error':
            return MeanAbsoluteError(reduction="auto", name="mean_absolute_error")
        elif loss == 'Mean Absolute Percentage Error':
            return MeanAbsolutePercentageError(reduction="auto", name="mean_absolute_percentage_error")
        elif loss == 'Mean Squared Logarithmic Error':
            return MeanSquaredLogarithmicError(reduction="auto", name="mean_squared_logarithmic_error")
        else:  # Default MSE
            return MeanSquaredError(reduction="auto", name="mean_squared_error")

    def create_model_from_params(self):

        self.model = Sequential()
        # LAYER 1
        if self.layers_type[0] == 'LSTM':
            self.input_shape = (self.n_timesteps, self.n_features)
            self.input_shape_onnx = [None, self.n_timesteps, self.n_features]
            self.model.add(LSTM(self.layers[0], input_shape=self.input_shape, recurrent_dropout=self.layers_dropout[0], activation=self.layers_activation_func[0]))
        elif self.layers_type[0] == 'Convolutional':
            self.input_shape = (self.n_features, 1)
            self.input_shape_onnx = [None, self.n_features]
            self.model.add(Conv1D(filters=self.layers[0], kernel_size=3, input_shape=self.input_shape,  activation=self.layers_activation_func[0]))
            self.model.add(MaxPooling1D(pool_size=2))
        else:
            self.input_shape = (self.n_features,)
            self.input_shape_onnx = [None, self.n_features]
            self.model.add(Dense(self.layers[0], input_shape=self.input_shape, activation=self.layers_activation_func[0]))
        if self.layers_dropout[0] > 0:
            self.model.add(Dropout(self.layers_dropout[0]))

        # HIDDEN LAYERS
        for l in range(1, len(self.layers)):
            if self.layers_type[l] == 'Dense':
                self.model.add(Dense(self.layers[l], activation=self.layers_activation_func[l]))
            elif self.layers_type[l] == 'LSTM':
                self.model.add(LSTM(self.layers[l], activation=self.layers_activation_func[l], recurrent_dropout=self.layers_dropout[l]))
            else:  # self.layers_type[l] == 'CNN':
                self.model.add(Conv1D(self.layers[l], kernel_size=3, activation=self.layers_activation_func[l]))
            if self.layers_dropout[l] > 0:
                self.model.add(Dropout(self.layers_dropout[l]))

        # OUTPUT LAYER
        self.model.add(Dense(1, activation=None))

        return

    def create_training_callbacks(self):
        callbacks = []
        if self.patience > 0:
            es = EarlyStopping(monitor='loss', mode='min', verbose=0, patience=self.patience)
            callbacks.append(es)
        mc = ModelCheckpoint(f'models/{self.name}/best_model.h5', monitor='loss', mode='min', verbose=1, save_best_only=True)
        callbacks.append(mc)

        # callbacks.append(StreamlitCallback(self.epochs))
        # callbacks.append(StdoutCallback(self.epochs))
        callbacks.append(DBoutCallback(self.epochs))
        return callbacks

    def reshape_input_lstm(self, x):
        columns = [x.shift(i) for i in range(self.n_timesteps)]
        x = concat(columns, axis=1)
        x = x.values.reshape(x.shape[0], self.n_timesteps, self.n_features)
        return x[self.n_timesteps-1:]  # remove fist null entries

    @staticmethod
    def reshape_input_cnn(x):
        return None  # remove fist null entries

    def train(self, x_train, y_train, x_val=None, y_val=None):

        callbacks = self.create_training_callbacks()

        validation_data = (x_val, y_val)

        # FIT NETWORK
        start = time.time()
        comp = self.model.compile(loss='mse', optimizer='adam', metrics=[self.loss_function])

        # TRANSFORM INPUTS IF NECESSARY
        if self.layers_type[0] == "LSTM":
            history = self.model.fit(self.reshape_input_lstm(x_train), y_train[self.n_timesteps - 1:].values,
                                     epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose,
                                     callbacks=callbacks)
        else:
            history = self.model.fit(x_train.values, y_train.values, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose, callbacks=callbacks)
        end = time.time()
        self.history = history
        # print('Total Training Time', str(round((end-start), 2))+'s')
        return history

    def get_eval_scores(self, y_test, y_pred, scaler):
        if scaler is not None:
            y_test, y_pred = reverse_scale_data(y_test, y_pred, scaler)
        return (y_test, y_pred, mean_absolute_error(y_test, y_pred), sqrt(mean_squared_error(y_test, y_pred)),
                max_error(y_test, y_pred), self.relative_absolute_error(y_test, y_pred), r2_score(y_test, y_pred))

    def evaluate_model(self, x_test, y_test, scaler):
        if self.layers_type[0] == "LSTM":
            x_test = self.reshape_input_lstm(x_test)
            y_pred = self.model.predict(x_test).reshape(-1)
            return self.get_eval_scores(y_test[self.n_timesteps - 1:], y_pred, scaler)
        else:
            y_pred = self.model.predict(x_test).reshape(-1)
            return self.get_eval_scores(y_test, y_pred, scaler)

    def get_model(self):
        return self.model

    def get_training_history(self):
        return self.history

    def get_plot_training_results(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))

        # ax1.plot(self.history.history['accuracy'], label='Train Accuracy')
        # ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        # ax1.grid(axis='y', linestyle='--', linewidth=0.5)
        # ax1.legend()

        ax1.plot(self.history.history['loss'], label='Train Loss')
        ax2.plot(self.history.history['mean_squared_error'], label='Mean Squared Error')
        ax1.grid(axis='y', linestyle='--', linewidth=0.5)
        ax1.legend()
        return fig

    @staticmethod
    def relative_absolute_error(true, pred):
        true_mean = np_mean(true)
        squared_error_num = np_sum(np_abs(true - pred))
        squared_error_den = np_sum(np_abs(true - true_mean))
        rae_loss = squared_error_num / squared_error_den
        return rae_loss

    def get_model_insights(self):
        pyplot(self.get_plot_training_results(), use_container_width=True)
        return

    def evaluate(self, x_test, y_test, verbose=0, batch_size=16, result_log=0):
        self.evaluation_score = self.model.evaluate(x_test, y_test, batch_size=batch_size, verbose=verbose)
        if result_log:
            print("%s: %.2f%%" % (self.model.metrics_names[1], self.evaluation_score[1]*100))
        return self.evaluation_score

    def export_model_keras(self):
        try:
            self.model.save(MODEL_SAVE_TEMP_PATH + 'ml_model.keras')
            return 1, None
        except Exception as e:
            print('DNN Export to Keras failed', e)
            return 0, 'Model Exportation to .keras failed.'

    def export_model_h5(self):
        try:
            self.model.save(MODEL_SAVE_TEMP_PATH + 'ml_model.h5')
            return 1, None
        except Exception as e:
            print('DNN Export to h5 failed', e)
            return 0, 'Model Exportation to .h5 failed.'

    @staticmethod
    def export_model_pickle():
        return 0, 'Pickle is not supported for the current model. Choose **.keras** or **.h5**.'

    @staticmethod
    def export_model_joblib():
        return 0, 'Joblib is not supported for the current model. Choose **.keras** or **.h5**.'

    def export_model_onnx(self):
        res = get_onnx_model(self.model, self.input_shape_onnx)
        return res

    def save_model(self, path_dir):
        self.model.save(path_dir+'model.keras')
        return

    def load_model(self, path_dir):
        self.model = load_model(path_dir+'model.keras')
        return
