#  streamlit as st # from streamlit import session_state
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from seaborn import scatterplot, lineplot
from plotly.express import line, scatter
from numpy import linspace
from plotly.express.colors import sequential
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def _plot_regression(df, degree, rx, ry, rz, bins=11, x_label='', y_label='', min_samples=15, bias=False, figx=8, figy=8):
    # x_label = rx; y_label = ry # axis Labels
    # rx = stop_dict(rx); ry = stop_dict(ry); rz = stop_dict(rz) # get actual feature name in dataframe
    poly = PolynomialFeatures(degree=degree, include_bias=bias, interaction_only=False)
    poly_reg_model = LinearRegression()
    reg_res = []
    reg_res_y = []
    reg_res_p = []
    formulas = []
    # print('f(x) ->', ry, ' x ->', rx)
    for i in range(1, bins):
        df_tmp = df[df[rz] == i]
        X = df_tmp[rx].values
        y = df_tmp[ry].values
        if len(X) >= min_samples:
            poly_features = poly.fit_transform(X.reshape(-1, 1))
            poly_reg_model.fit(poly_features, y)
            y_predicted = poly_reg_model.predict(poly_features)
            reg_res.extend(X)
            reg_res_y.extend(y_predicted)
            reg_res_p.extend([i] * len(X))
            x_name = 'load'
            if degree == 1:
                # print('Formula for', rz, 'bin', str(i) + ':', ry, '=', poly_reg_model.intercept_, '+', str(poly_reg_model.coef_[0]) + '*x')
                formula = 'Bin\ ' + str(i) +  ' \\rightarrow ' + ry + ' = ' + str(round(poly_reg_model.intercept_, 3)) + ' + ' + str(round(poly_reg_model.coef_[0], 3)) + '\cdot '+x_name #
                # r = r''' s''' + r''' * ''' + r''' d '''
                formulas.append(formula)
            elif degree == 2:
                # print('Formula for', rz, 'bin', str(i) + ':', ry, '=', poly_reg_model.intercept_, '+', str(poly_reg_model.coef_[0]) + '*x', '+', str(poly_reg_model.coef_[1]) + '*x^2')
                formula = 'Bin\ ' + str(i) + ' \\rightarrow ' + ry + ' = ' + str(round(poly_reg_model.intercept_, 3)) + ' + ' + str(round(poly_reg_model.coef_[0], 3)) + '\cdot '+x_name +\
                          ' + ' + str(round(poly_reg_model.coef_[1], 3)) + '\cdot ' +x_name + '^2' #
                formulas.append(formula)
            elif degree == 3:
                # print('Formula for', rz, 'bin', str(i) + ':', ry, '=', poly_reg_model.intercept_, '+', str(poly_reg_model.coef_[0]) + '*x', '+', str(poly_reg_model.coef_[1]) + '*x^2', '+', str(poly_reg_model.coef_[2]) + '*x^3')
                formula = 'Bin\ ' + str(i) + ' \\rightarrow ' + ry + ' = ' + str(round(poly_reg_model.intercept_, 3)) + ' + ' + str(round(poly_reg_model.coef_[0], 3)) + '\cdot ' +x_name +\
                          ' + ' + str(round(poly_reg_model.coef_[1], 3)) + '\cdot ' +x_name+'^2' + ' + ' + str(round(poly_reg_model.coef_[2], 3)) + '\cdot ' +x_name+'^3'
                formulas.append(formula)
            elif degree == 4:
                formula = 'Bin\ ' + str(i) + ' \\rightarrow ' + ry + ' = ' + str(
                    round(poly_reg_model.intercept_, 3)) + ' + ' + str(round(poly_reg_model.coef_[0], 3)) + '\cdot x' + \
                          ' + ' + str(round(poly_reg_model.coef_[1], 3)) + '\cdot x^2' + ' + ' + str(
                    round(poly_reg_model.coef_[2], 3)) + '\cdot x^3' ' + ' + str(
                    round(poly_reg_model.coef_[4], 3)) + '\cdot x^4'
                formulas.append(formula)
            elif degree == 5:
                pass
            else:
                pass
    df_reg = DataFrame(reg_res, columns=['x'])
    df_reg['y'] = reg_res_y
    df_reg[rz] = reg_res_p
    fig, ax = plt.subplots(figsize=(figx, figy))
    # print('Training Done')
    # sns.set_theme(style="whitegrid")
    scatterplot(data=df_reg, x='x', y='y', hue=rz, palette='coolwarm', legend=False, marker='x', ax=ax)
    plot = lineplot(data=df_reg, x='x', y='y', hue=rz, palette='coolwarm', linewidth=3, ax=ax)
    plot.set_xlabel(x_label)
    plot.set_ylabel(y_label)
    return fig, formulas



def _plot_regression_regplot(df, degree, rx, ry, rz, bins=6, x_label="", y_label="", min_samples=15, bias=False, outlier_removal=True):
    poly = PolynomialFeatures(degree=degree, include_bias=bias, interaction_only=False)
    poly_reg_model = LinearRegression()
    reg_res = []
    reg_res_y = []
    reg_res_p = []
    formulas = []
    for i in range(1, bins):
        df_tmp = df[df[rz] == i]
        X = df_tmp[rx].values
        y = df_tmp[ry].values

        if len(X) >= min_samples:

            if outlier_removal:
                p5 = np.percentile(y, 5)
                p95 = np.percentile(y, 95)
                # Filter out values above the 95th percentile and below the 5th percentile
                y = y[(y >= p5) & (y <= p95)]
                X = X[(y >= p5) & (y <= p95)]

            poly_features = poly.fit_transform(X.reshape(-1, 1))
            poly_reg_model.fit(poly_features, y)
            y_predicted = poly_reg_model.predict(poly_features)

            reg_res.extend(X)
            reg_res_y.extend(y_predicted)
            reg_res_p.extend([i] * len(X))
            x_name = 'load'
            if degree == 1:
                formula = 'Bin\ ' + str(i) +  ' \\rightarrow ' + ry + ' = ' + str(round(poly_reg_model.intercept_, 3)) + ' + ' + str(round(poly_reg_model.coef_[0], 3)) + '\cdot '+x_name #
                formulas.append(formula)
            elif degree == 2:
                formula = 'Bin\ ' + str(i) + ' \\rightarrow ' + ry + ' = ' + str(round(poly_reg_model.intercept_, 3)) + ' + ' + str(round(poly_reg_model.coef_[0], 3)) + '\cdot '+x_name +\
                          ' + ' + str(round(poly_reg_model.coef_[1], 3)) + '\cdot ' +x_name + '^2' #
                formulas.append(formula)
            elif degree == 3:
                formula = 'Bin\ ' + str(i) + ' \\rightarrow ' + ry + ' = ' + str(round(poly_reg_model.intercept_, 3)) + ' + ' + str(round(poly_reg_model.coef_[0], 3)) + '\cdot ' +x_name +\
                          ' + ' + str(round(poly_reg_model.coef_[1], 3)) + '\cdot ' +x_name+'^2' + ' + ' + str(round(poly_reg_model.coef_[2], 3)) + '\cdot ' +x_name+'^3'
                formulas.append(formula)
            elif degree == 4:
                formula = 'Bin\ ' + str(i) + ' \\rightarrow ' + ry + ' = ' + str(
                    round(poly_reg_model.intercept_, 3)) + ' + ' + str(round(poly_reg_model.coef_[0], 3)) + '\cdot x' + \
                          ' + ' + str(round(poly_reg_model.coef_[1], 3)) + '\cdot x^2' + ' + ' + str(
                    round(poly_reg_model.coef_[2], 3)) + '\cdot x^3' ' + ' + str(
                    round(poly_reg_model.coef_[4], 3)) + '\cdot x^4'
                formulas.append(formula)
            elif degree == 5:
                pass
            else:
                pass
    df_reg = DataFrame(reg_res, columns=['x'])
    df_reg['y'] = reg_res_y
    df_reg[rz] = reg_res_p
    from seaborn.regression import regplot
    from numpy import mean as np_mean

    diff_bins = len(df_reg[rz].unique())
    fig, axes = plt.subplots(1, diff_bins, figsize=(18, 6))
    if diff_bins > 1:
        axes = axes.flatten()
    for i in range(diff_bins):
        axes[i].set_title('Dependence Plot')
        regplot(x='x', y='y', data=df_reg[df_reg[rz] == i+1], ax=axes[i], fit_reg=True, order=3,
                x_bins=500, x_estimator=np_mean, ci=95, color="red",
                scatter_kws={"s": 20, "color": "blue", 'alpha': 0.9, "marker": "o"},
                line_kws={"color": "red", "linewidth": 2})

    return fig, formulas


# from pygam import LinearGAM, s, f
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
#
# def _plot_regression_gamsel(df, degree, rx, ry, rz, bins=6, x_label="", y_label="", min_samples=15, bias=False, outlier_removal=True):
#
#     reg_res = []
#     reg_res_y = []
#     reg_res_y_hi = []
#     reg_res_y_lo = []
#     reg_res_p = []
#
#     for i in range(1, bins):
#         df_tmp = df[df[rz] == i]
#         X = df_tmp[rx].values
#         y = df_tmp[ry].values
#         if len(X) >= min_samples:
#
#             gam = LinearGAM(n_splines=25).gridsearch(X.reshape(-1,1), y)
#             XX = gam.generate_X_grid(term=0, n=len(X))
#             gam_ci = gam.prediction_intervals(XX, width=.95)
#
#             reg_res.extend(X)
#             reg_res_y.extend(gam.predict(XX))
#             reg_res_y_lo.extend([i[0] for i in gam_ci])
#             reg_res_y_hi.extend([i[1] for i in gam_ci])
#             reg_res_p.extend([i] * len(X))
#
#
#     df_reg = DataFrame(reg_res, columns=['x'])
#     df_reg['y'] = reg_res_y
#     df_reg['y_lo'] = reg_res_y_lo
#     df_reg['y_hi'] = reg_res_y_hi
#     df_reg[rz] = reg_res_p
#
#     fig = go.Figure()
#
#     j = []
#     for i in XX:
#         j.append(i[0])
#     reg_res = j
#     fig.add_trace(go.Scatter(x=reg_res, y=reg_res_y))
#     fig.add_trace(go.Scatter(x=reg_res, y=reg_res_y_lo))
#     fig.add_trace(go.Scatter(x=reg_res, y=reg_res_y_hi))
#
#     return fig, ['dsd', 'sdsd']






# from sklearn.ensemble import GradientBoostingRegressor
#  LOWER_ALPHA = 0.1
#  UPPER_ALPHA = 0.9
#  # Each model has to be separate
#  lower_model = GradientBoostingRegressor(loss="quantile",
#                                          alpha=LOWER_ALPHA)
#  # The mid model will use the default loss
#  mid_model = GradientBoostingRegressor(loss="ls")
#  upper_model = GradientBoostingRegressor(loss="quantile",
#                                          alpha=UPPER_ALPHA)
#  lower_model.fit(X.reshape(-1, 1), y)
#  mid_model.fit(X.reshape(-1, 1), y)
#  upper_model.fit(X.reshape(-1, 1), y)
#
#  X_test = np.linspace(0,1.05, 1000)
#  y_predicted_lo = lower_model.predict(X_test.reshape(-1, 1))
#  y_predicted_mid = mid_model.predict(X_test.reshape(-1, 1))
#  y_predicted_hi = upper_model.predict(X_test.reshape(-1, 1))
#
#  reg_res.extend(X_test)
#  reg_res_y.extend(y_predicted_mid)
#  reg_res_p.extend([i] * len(X_test))
#  x_name = 'load'
#
#