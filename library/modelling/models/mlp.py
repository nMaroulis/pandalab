from sklearn.neural_network import MLPRegressor
from streamlit import pyplot, write, spinner
import matplotlib.pyplot as plt
from library.modelling.models.base_model import BaseModel


class MLPModel(BaseModel):

    def __init__(self, params=None, training_params=None):
        super().__init__("Feed-Forward Neural Network", MLPRegressor(**params), params, training_params)

    def training_results(self):
        write('Best Validation Score: ' + str(self.model.best_validation_score_))
        col_count = 0
        if self.model.best_loss_ is not None:
            col_count += 1
        if self.model.loss_curve_ is not None:
            col_count += 1
        if self.model.validation_scores_ is not None:
            col_count += 1
        fig, axs = plt.subplots(1, col_count, figsize=(16, 4))

        col_count = 0
        if self.model.best_loss_ is not None:
            axs[col_count].plot(self.model.best_loss_)
            axs[col_count].set_title('Best Loss')
            axs[col_count].grid(True)
            col_count += 1

        if self.model.loss_curve_ is not None:
            axs[col_count].plot(self.model.loss_curve_)
            axs[col_count].set_title('Loss Curve')
            axs[col_count].grid(True)
            col_count += 1
        if self.model.validation_scores_ is not None:
            axs[col_count].plot(self.model.validation_scores_)
            axs[col_count].set_title('Validation Scores')
            axs[col_count].grid(True)
        return fig

    def get_model_insights(self):
        write('The following showcase the progress of the Neural Network for every Epoch (Step) of the Training Process')
        with spinner("Generating Training Evaluation Results"):
            pyplot(self.training_results(), use_container_width=True)
