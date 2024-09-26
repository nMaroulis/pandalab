from sklearn.linear_model import LinearRegression, QuantileRegressor, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from streamlit import write, session_state, latex

from library.modelling.models.base_model import BaseModel


class LinearRegressionModel(BaseModel):

    def __init__(self, params=None, training_params=None):

        degree = params.get('degree')
        intercept = params.get('intercept')
        norm_coeffs = params.get('norm_coeffs')
        alpha = params.get('alpha')

        if norm_coeffs:
            super().__init__("Multinomial Ridge Regression", Ridge(alpha=alpha, fit_intercept=intercept), params, training_params)
        else:
            super().__init__("Multinomial Linear Regression", LinearRegression(fit_intercept=intercept), params, training_params)

        self.polynomial_transformer = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)

    def train(self, x_train, y_train):  # OVERRIDE
        self.n_features = len(x_train.columns)
        poly_x = self.polynomial_transformer.fit_transform(x_train)
        self.model.fit(poly_x, y_train)

    def generate_regression_formula(self):
        coefs = self.model.coef_
        intercept = round(self.model.intercept_, 4)
        final_formula = 'y = ' + str(intercept) + ' + '
        coef_counter = 0
        for p in self.polynomial_transformer.powers_:
            p_counter = 0
            coeff = round(coefs[coef_counter], 4)
            if coeff > 0:
                final_formula += str(coeff) + '*'
                for v in p:
                    if v > 0:
                        final_formula += 'x' + str(p_counter + 1) + '^' + str(v) if v > 1 else 'x' + str(p_counter + 1)
                    p_counter += 1
                final_formula += ' + '
            coef_counter += 1
        final_formula = final_formula[0:-3]
        return final_formula

    def evaluate_model(self, x_test, y_test, scaler):

        poly_x = self.polynomial_transformer.fit_transform(x_test)  # reshape X to polynomial  # .reshape(-1, 1)
        y_pred = self.model.predict(poly_x)

        return self.get_eval_scores(y_test, y_pred, scaler)

    def get_model_insights(self):
        write('The Linear Regression Model is trained in order to generate the best formula based on the'
              'independent variables to optimally estimate the Target variable. The **Coefficients** can later be'
              'extracted from the trained model in order to generate this formula')

        write('The final Regression formula is shown below, where:')
        col_counter = 1
        for f in session_state['input_features_list']:
            write(f'- **x{col_counter}** → **{f}**')
            col_counter += 1

        latex(self.generate_regression_formula())
