from xgboost import XGBRegressor
from re import compile, IGNORECASE
from streamlit import session_state, pyplot, write,spinner
from matplotlib.pyplot import subplots as plt_subplots
from numpy import array as np_array
from library.modelling.models.base_model import BaseModel


class XGBoostModel(BaseModel):

    def __init__(self, params=None, training_params=None):
        super().__init__("XGBoost", XGBRegressor(**params), params, training_params)

    def train(self, x_train, y_train):
        self.n_features = len(x_train.columns)
        regex = compile(r"\[|\]|<", IGNORECASE)
        x_train.columns = [regex.sub("", col) if any(i in str(col) for i in set(('[', ']', '<'))) else col for col in
                           x_train.columns.values]
        self.model.fit(x_train, y_train)

    def evaluate_model(self,x_test, y_test, scaler):

        regex = compile(r"\[|\]|<", IGNORECASE)
        x_test.columns = [regex.sub("", col) if any(i in str(col) for i in set(('[', ']', '<'))) else col for col in
                          x_test.columns.values]

        y_pred = self.model.predict(x_test)
        print(len(y_test), len(y_pred))
        self.get_eval_scores(y_test, y_pred, scaler)
        return self.get_eval_scores(y_test, y_pred, scaler)

    def feature_importance(self):
        columns = np_array(session_state['input_features_list'])

        sorted_idx = self.model.feature_importances_.argsort()
        height_size = len(columns) // 3
        if height_size < 2:
            height_size = 2
        fig, ax = plt_subplots(1, 1, figsize=(20, height_size))
        ax.barh(columns[sorted_idx], self.model.feature_importances_[sorted_idx]*100)
        ax.set_xlabel("Feature Importance for the ML Model (%)")
        ax.grid(axis='y')
        return fig

    def get_model_insights(self):
        write('XGBoost consists of many decision trees. Across each tree, the number of times a feature is used to'
              ' split the data. If a feature is used a lot, it has a high **weight**. Therefore the **Weight** is '
              'translated as the **Feature Importance**, meaning **how important was each Feature for the estimation '
              'of the Target**.')
        with spinner("Generating Feature Importance"):
            pyplot(self.feature_importance(), use_container_width=True)
