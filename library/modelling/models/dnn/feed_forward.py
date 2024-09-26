from tensorflow.keras.layers import Dense, Flatten, Dropout, LSTM, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import L1, L2
from library.modelling.models.dnn.dnn_model import DNNModel
from time import time

class FeedForwardModel(DNNModel):

    def __init__(self, params=None, n_features=None):
        super().__init__(params, n_features, 'FeedForward')
        self.model = self.load_default_model_sm()

    def load_default_model_sm(self):
        model = Sequential()
        # LAYER 1
        model.add(Dense(64, input_shape=(self.n_features,), activation='relu'))
        model.add(Dropout(0.2))
        # LAYER 2
        model.add(Dense(32, activation='relu', activity_regularizer=L2(0.001)))
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu', activity_regularizer=L2(0.001)))
        # OUTPUT LAYER
        model.add(Dense(1, activation=None))
        return model

    def train(self, x_train, y_train, x_val=None, y_val=None):

        callbacks = self.create_training_callbacks()
        validation_data = (x_val, y_val)
        # FIT NETWORK
        start = time()
        comp = self.model.compile(loss='mse', optimizer='adam', metrics=[self.loss_function])
        history = self.model.fit(x_train.values, y_train.values, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose, callbacks=callbacks)
        end = time()
        self.history = history
        print('Total Training Time', str(round((end-start), 2))+'s')
        return history

    def evaluate_model(self, x_test, y_test, scaler):
        y_pred = self.model.predict(x_test).reshape(-1)
        return self.get_eval_scores(y_test, y_pred, scaler)
