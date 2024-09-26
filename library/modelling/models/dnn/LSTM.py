from tensorflow.keras.layers import Dense, Flatten, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import L1, L2
import time
from library.modelling.models.dnn.dnn_model import DNNModel


class LSTMModel(DNNModel):

    def __init__(self, params=None, n_features=None):
        super().__init__(params, n_features, 'LSTM')
        self.n_timesteps = params.get("n_timesteps")
        self.model = self.load_default_model_sm()

    def load_default_model_sm(self):
        model = Sequential()
        model.add(LSTM(100, input_shape=(self.n_timesteps, self.n_features), recurrent_dropout=0.2))
        model.add(Dense(64, activation='relu', activity_regularizer=L2(0.001)))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu', activity_regularizer=L2(0.001)))
        model.add(Dense(1, activation=None))
        return model

    def train(self, x_train, y_train, x_val=None, y_val=None):

        callbacks = self.create_training_callbacks()
        validation_data = (x_val, y_val)
        # FIT NETWORK
        start = time.time()
        comp = self.model.compile(loss='mse', optimizer='adam', metrics=[self.loss_function])
        history = self.model.fit(self.reshape_input_lstm(x_train), y_train[self.n_timesteps-1:].values, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose, callbacks=callbacks)
        end = time.time()
        self.history = history
        print('Total Training Time', str(round((end-start), 2))+'s')
        return history

    def evaluate_model(self, x_test, y_test, scaler):
        x_test = self.reshape_input_lstm(x_test)
        y_pred = self.model.predict(x_test).reshape(-1)
        return self.get_eval_scores(y_test[self.n_timesteps-1:], y_pred, scaler)


"""
Classification Functions
    def plot_confusion_matrix(self, x_test, y_test, display_labels=None):
        yp = self.model.predict(x_test)
        cf_matrix = confusion_matrix(y_test, argmax(yp, axis=-1))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
        heatmap(cf_matrix/sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues', ax=ax1)
        ax1.set_title('Confusion Matrix - Percentage\n\n')
        ax1.set_xlabel('\nPredicted Values'); ax1.set_ylabel('Actual Values ')
        ax1.xaxis.set_ticklabels(display_labels); ax1.yaxis.set_ticklabels(display_labels)
        heatmap(cf_matrix, annot=True, fmt='d', cmap='YlOrBr', ax=ax2)
        ax2.set_title('Confusion Matrix - Absolute\n\n')
        ax2.set_xlabel('\nPredicted Values'); ax2.set_ylabel('Actual Values ')
        ax2.xaxis.set_ticklabels(display_labels); ax2.yaxis.set_ticklabels(display_labels)
        plt.tight_layout(); plt.show()
        return

    def get_false_predictions(self, pred_class, true_label):
        falsely_classified = []
        for i in range(len(pred_class)):
            if pred_class[i] != true_label:
                falsely_classified.append([i, pred_class[i]])
        return falsely_classified
        
    def predict(self, x_test, get_prob=False, classes=None, class_labels=None):
        yp = self.model.predict(x_test)
        pred_class = argmax(yp, axis=-1)
        if classes is None or class_labels is None or (len(classes) != len(class_labels)):
            pass
        else:
            print('Classified as:')
            for i in range(len(class_labels)):
                n = count_nonzero(pred_class == class_labels[i])
                print('-', str(classes[i]) + ':', n, '/', str(round((n/yp.shape[0])*100,2))+'%')
        if get_prob:
            return yp
        else:
            return pred_class
            
def get_last_evaluation_score(self):
    return self.evaluation_score
"""