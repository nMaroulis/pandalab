from streamlit import session_state, markdown, info, expander, write, form, form_submit_button, spinner, radio,\
    selectbox, data_editor, pyplot, plotly_chart, columns, slider, code, warning, latex, toggle, fragment
from mpld3 import fig_to_html
from streamlit.components.v1 import html as components_html
import matplotlib.pyplot as plt
from seaborn import regplot, color_palette, lmplot
from numpy import linspace, quantile, mean as np_mean, inf as np_inf, nan as np_nan
from pandas import DataFrame, cut  # qcut
from plotly.express import line
from plotly.graph_objects import Scatter, Figure, Histogram2dContour
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, QuantileRegressor, Ridge
from sklearn.preprocessing import PolynomialFeatures


def dependence_plot(x, y, degree=3):
    fig, ax = plt.subplots(1, 1, figsize=(18, 6))

    REG = True
    ORDER = degree
    REG_COLOUR = "red"
    S_SIZE = 20
    S_COLOUR = "blue"
    S_ALPHA = 0.9
    CI = 95
    X_BINS = 500
    S_MARKER = "o"
    L_COLOUR = "red"
    L_WIDTH = 2

    ax.set_title('Dependence Plot')  # axes[i][j].set_title(name)
    regplot(x=x, y=y, data=session_state.df[[x, y]].replace([np_inf, -np_inf], np_nan).dropna(), ax=ax, fit_reg=REG, order=ORDER, x_bins=X_BINS, x_estimator=np_mean, ci=CI,
            color=REG_COLOUR, scatter_kws={"s": S_SIZE, "color": S_COLOUR, 'alpha': S_ALPHA, "marker": S_MARKER},
            line_kws={"color": L_COLOUR, "linewidth": L_WIDTH})
    return fig


def dependence_plot_3d(x, y, z, bins=4, degree=3):

    REG = True
    ORDER = degree
    CI = 95
    X_BINS = 400
    z_bin = z + '_bin'
    session_state.df[z_bin] = cut(session_state.df[z], bins=bins)
    # session_state.df[z_bin] = qcut(session_state.df[z], q=bin_count)
    ub = list(session_state.df[z_bin].unique().dropna())
    ub.sort()
    unique_bins = []
    for i in range(len(ub)):
        unique_bins.append('Bin ' + str(i+1) + ': ' + str(ub[i]))
    lm = lmplot(x=x, y=y, hue=z_bin, data=session_state.df[[x, y, z, z_bin]].replace([np_inf, -np_inf], np_nan).dropna(), fit_reg=REG, order=ORDER,
                x_bins=X_BINS, x_estimator=np_mean, ci=CI, height=5, aspect=2, legend=False
              )
    session_state.df.drop(z_bin, inplace=True, axis=1)
    return lm.fig, unique_bins, color_palette("tab10").as_hex()


def dependence_plot_3d_subplots(x, y, z, bins=4, degree=3):

    REG = True
    ORDER = degree
    REG_COLOUR = "red"
    S_SIZE = 12
    S_COLOUR = "blue"
    S_ALPHA = 0.9
    CI = 95
    X_BINS = 200
    S_MARKER = "o"
    L_COLOUR = "red"
    L_WIDTH = 2

    z_bin = z + '_bin'
    session_state.df[z_bin] = cut(session_state.df[z], bins=bins)
    ub = list(session_state.df[z_bin].unique().dropna())
    ub.sort()
    bin_count = len(ub)
    if bin_count == 2:
        fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    elif bin_count == 3:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    elif bin_count == 4:
        fig, axes = plt.subplots(1, 4, figsize=(18, 6), sharey=True)
    elif bin_count == 5 or bin_count ==6:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharey=True)
    else: #  bin_count >= 7
        fig, axes = plt.subplots(2, 4, figsize=(18, 12), sharey=True)
    axes = axes.flatten()
    for i in range(len(ub)):
        axes[i].set_title(z + ': ' + str(ub[i])).set_fontsize(10)  # axes[i][j].set_title(name)
        regplot(x=x, y=y, data=session_state.df[session_state.df[z_bin] == ub[i]][[x, y, z]].replace([np_inf, -np_inf], np_nan).dropna(), ax=axes[i], fit_reg=REG, order=ORDER,
                x_bins=X_BINS, x_estimator=np_mean, ci=CI, color=REG_COLOUR,
                scatter_kws={"s": S_SIZE, "color": S_COLOUR, 'alpha': S_ALPHA, "marker": S_MARKER},
                line_kws={"color": L_COLOUR, "linewidth": L_WIDTH})

    session_state.df.drop(z_bin, inplace=True, axis=1)
    # fig.tight_layout()
    return fig


def dependence_plot_3d_trisurf(x, y, z, bin_count=4):
    z_bin = z + '_bin'
    session_state.df[z_bin] = cut(session_state.df[z], bins=bin_count, labels=list(range(1, bin_count+1)))

    fig = plt.figure(figsize=(16, 16))
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(session_state.df[x], session_state.df[y], session_state.df[z_bin], cmap='coolwarm',
                    edgecolor='none', antialiased=True)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z_bin)
    return fig

### -------------- KNN -----------------------------

def get_knn_neighbors(df_shape = 1000):
    if df_shape >= 1000000:
        return 111
    elif df_shape >= 100000:
        return 81
    elif df_shape >= 10000:
        return 31
    elif df_shape >= 1000:
        return 9
    elif df_shape >= 500:
        return 5
    else:
        return 3


def dependence_plot_plotly_knn(x, y, show_scatter=True):

    df = session_state.df[[x,y]].replace([np_inf, -np_inf], np_nan).dropna()
    X = df[x].values.reshape(-1, 1)
    x_range = linspace(quantile(X, 0.01), quantile(X, 0.98), 100)
    knn_uni = KNeighborsRegressor(get_knn_neighbors(df.shape[0]), weights='uniform')
    knn_uni.fit(X, df[y])
    y_uni = knn_uni.predict(x_range.reshape(-1, 1))

    r_squared = knn_uni.score(X, df[y])
    fig = Figure()
    fig.add_traces(Scatter(x=x_range, y=y_uni, name=y, mode='lines', line=dict(color='rgba(255, 99, 71,1)',width=2)))

    if show_scatter:
        if len(X) > 2000:
            skipping_step = int(len(X) / 2000)
        else:
            skipping_step = 1
        fig.add_traces(Scatter(x=X[1::skipping_step], y=df[y][1::skipping_step].values, marker_symbol='x', mode='markers', marker_color='rgba(135, 206, 255, 0.5)', hoverinfo='skip', name='Data Points'))

    fig.update_xaxes(title_text=x)
    fig.update_yaxes(title_text=y)

    return fig, r_squared


# Fast 3D Plot [Subplots] with KNN
def dependence_plot_plotly_knn_3d(x, y, z, bins, type="subplots"):
    df = session_state.df[[x, y, z]].replace([np_inf, -np_inf], np_nan).dropna()

    z_bin = z + '_bin'
    df[z_bin] = cut(df[z], bins=bins)
    ub = list(df[z_bin].unique().dropna())
    ub.sort()
    reg_res = []
    reg_res_y = []
    reg_res_p = []
    r_squared_results = []
    for i in range(len(ub)):
        df_tmp = df[df[z_bin] == ub[i]]

        if df_tmp.shape[0] >= 3:

            X = df_tmp[x].values.reshape(-1, 1)
            x_range = linspace(quantile(X, 0.01), quantile(X, 0.98), 100) # linspace(quantile(X, 0.01), quantile(X, 0.98), 100)
            knn_uni = KNeighborsRegressor(get_knn_neighbors(df_tmp.shape[0]), weights='uniform')
            knn_uni.fit(X, df_tmp[y])
            y_uni = knn_uni.predict(x_range.reshape(-1, 1))

            r_squared = knn_uni.score(X, df_tmp[y])
            r_squared_results.append("For **" + z + "** value range of **" + str(ub[i]) + "**: ***~" + str(int(round(r_squared, 3) * 100)) + "% of the variance of " + y + " is explained by " + x + "*** in the KNN Regressor model.")

            reg_res.extend(x_range)
            reg_res_y.extend(y_uni)
            reg_res_p.extend([str(ub[i])] * len(x_range))
    df_reg = DataFrame(reg_res, columns=[x])
    df_reg[y] = reg_res_y
    df_reg[z_bin] = reg_res_p

    # fig = scatter(df_reg, x='x', y='y', facet_col=z_bin, marginal_x="box")
    if type == "subplots":
        fig = line(df_reg, x=x, y=y, facet_col=z_bin)
    else:  # single
        fig = line(df_reg, x=x, y=y, color=z_bin, line_group=z_bin)

    fig.update_layout(
        title=x + " and " + y + " relationship plot for each " + z + " bin.",
        xaxis_title=x,
        yaxis_title=y,
        legend_title=z + " bin",
    )
    del df[z_bin]
    return fig, r_squared_results


# Fast 2D Plot with Linear Regression
def get_regression_formula(poly_reg_model=None, degree=3):
    formula = ""
    if degree == 1:
        formula = ('y = ' + str(round(poly_reg_model.intercept_, 3)) + ' + ' +
                   str(round(poly_reg_model.coef_[0], 3)) + '\cdot x')
    elif degree == 2:
        formula = ('y = ' + str(round(poly_reg_model.intercept_, 3)) + ' + ' + str(round(poly_reg_model.coef_[0], 3))
                   + '\cdot x + ' + str(round(poly_reg_model.coef_[1], 3)) + '\cdot x^2')
    elif degree == 3:
        formula = ('y = ' + str(round(poly_reg_model.intercept_, 3)) + ' + ' + str(round(poly_reg_model.coef_[0], 3))
                   + '\cdot x + ' + str(round(poly_reg_model.coef_[1], 3)) + '\cdot x^2' + ' + ' +
                   str(round(poly_reg_model.coef_[2], 3)) + '\cdot x^3')
    elif degree == 4:
        formula = ('y = ' + str(round(poly_reg_model.intercept_, 3)) + ' + ' + str(round(poly_reg_model.coef_[0], 3))
                   + '\cdot x' + ' + ' + str(round(poly_reg_model.coef_[1], 3)) + '\cdot x^2' + ' + ' +
                   str(round(poly_reg_model.coef_[2], 3)) + '\cdot x^3' + ' + ' + str(round(poly_reg_model.coef_[3], 3)) +
                   '\cdot x^4')
    elif degree == 5:
        formula = ('y = ' + str(round(poly_reg_model.intercept_, 3)) + ' + ' + str(round(poly_reg_model.coef_[0], 3))
                   + '\cdot x' + ' + ' + str(round(poly_reg_model.coef_[1], 3)) + '\cdot x^2' + ' + ' +
                   str(round(poly_reg_model.coef_[2], 3)) + '\cdot x^3' + ' + ' + str(round(poly_reg_model.coef_[3], 3)) +
                   '\cdot x^4' + ' + ' +  str(round(poly_reg_model.coef_[4], 3)) + '\cdot x^5')
    else:
        pass
    return formula


def dependence_plot_plotly_lr(df, x, y, ci=False, degree=3, norm_coefs=False, incl_intercept=True, show_scatter=False):

    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)  # Polynomial Transformation
    if norm_coefs:
        lr_model = Ridge(fit_intercept=incl_intercept)  # Linear Regression Model
    else:
        lr_model = LinearRegression(fit_intercept=incl_intercept)  # Linear Regression Model

    df = df.replace([np_inf, -np_inf], np_nan).dropna()

    X = df[x].values  # X value
    Y = df[y].values  # y value

    poly_features = poly.fit_transform(X.reshape(-1, 1))  # reshape X to polynomial

    x_range = linspace(quantile(X, 0.01), quantile(X, 0.99), 100) # create a range of 100 point for plot, instead of min/max keep percentiles of X
    x_range_tr = poly.transform(x_range.reshape(-1, 1))  # transform plot x

    lr_model.fit(poly_features, Y)  # train model with reshaped X
    y_pred = lr_model.predict(x_range_tr)  # create presiction for plot X
    r_squared = lr_model.score(poly_features, Y)  # Get R-SQUARED
    if ci:
        lo_model = QuantileRegressor(quantile=0.01, solver='highs')  # TEST
        hi_model = QuantileRegressor(quantile=0.99, solver='highs')  # TEST
        lo_model.fit(poly_features, Y)  # TEST
        print('lo_model')
        hi_model.fit(poly_features, Y)  # TEST
        print('hi_model')
        y_lo_pred = lo_model.predict(x_range_tr)
        y_hi_pred = hi_model.predict(x_range_tr)

    fig = Figure()

    if show_scatter:
        if len(X) > 2000:
            skipping_step = int(len(X) / 2000)
        else:
            skipping_step = 1
        fig.add_traces(Scatter(x=X[1::skipping_step], y=Y[1::skipping_step], marker_symbol='x', mode='markers', marker_color='rgba(135, 206, 255, 0.5)', hoverinfo='skip', name='Data Points'))

    fig.add_traces(Scatter(x=x_range, y=y_pred, name=y, mode='lines', line=dict(color='rgba(255, 99, 71,1)',width=2)))


    if ci:
        fig.add_traces(Scatter(x=x_range, y=y_lo_pred, name=y+'_lower', mode='lines', line=dict(color='rgba(15, 99, 71,1)',width=2)))
        fig.add_traces(Scatter(x=x_range, y=y_hi_pred, name=y+'_upper', mode='lines', line=dict(color='rgba(15, 99, 71,1)',width=2)))

    fig.update_xaxes(title_text=x)
    fig.update_yaxes(title_text=y)

    formula = get_regression_formula(poly_reg_model=lr_model, degree=degree)

    return fig, r_squared, formula


def dependence_plot_plotly_lr_3d(df, x, y, z="", bins=4, ci=False, degree=3, norm_coefs=False, incl_intercept=True):

    df = df.replace([np_inf, -np_inf], np_nan).dropna()
    z_bin = z + '_bin'
    df[z_bin] = cut(df[z], bins=bins)
    ub = list(df[z_bin].unique().dropna())
    ub.sort()
    reg_res = []
    reg_res_y = []
    reg_res_p = []
    r_squared_results = []
    formulas = []
    for i in range(len(ub)):
        df_tmp = df[df[z_bin] == ub[i]]

        if df_tmp.shape[0] >= 5:

            X = df_tmp[x].values  # X value
            Y = df_tmp[y].values  # y value

            poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)  # Polynomial Transformation

            if norm_coefs:
                lr_model = Ridge(fit_intercept=incl_intercept)  # Linear Regression Model
            else:
                lr_model = LinearRegression(fit_intercept=incl_intercept)  # Linear Regression Model

            poly_features = poly.fit_transform(X.reshape(-1, 1))  # reshape X to polynomial

            x_range = linspace(quantile(X, 0.01), quantile(X, 0.98),
                               100)  # create a range of 100 point for plot, instead of min/max keep percentiles of X
            x_range_tr = poly.transform(x_range.reshape(-1, 1))  # transform plot x

            lr_model.fit(poly_features, Y)  # train model with reshaped X
            y_pred = lr_model.predict(x_range_tr)  # create presiction for plot X
            r_squared = lr_model.score(poly_features, Y)
            r_squared_results.append("For **" + z + "** value range of **" + str(ub[i]) + "**: ***~" + str(int(round(r_squared, 3) * 100)) + "% of the variance of " + y + " is explained by " + x + "*** in the Polynomlial Linear Regression model of degree "+str(3)+".")

            reg_res.extend(x_range)
            reg_res_y.extend(y_pred)
            reg_res_p.extend([str(ub[i])] * len(x_range))

            # Get Regression Formula
            formula = get_regression_formula(poly_reg_model=lr_model, degree=degree)
            formulas.append(str(ub[i]) + ' \\rightarrow ' + formula)
    df_reg = DataFrame(reg_res, columns=[x])
    df_reg[y] = reg_res_y
    df_reg[z_bin] = reg_res_p

    fig = line(df_reg, x=x, y=y, color = z_bin, line_group = z_bin) # color_discrete_sequence = sequential.YlOrRd

    fig.update_layout(
        title= x + " and " + y + " relationship plot for each " + z + " bin.",
        xaxis_title=x,
        yaxis_title=y,
        legend_title= z + " bin",
    )
    del df[z_bin]

    return fig, r_squared_results, formulas


def dependence_plot_plotly_lr_3d_subplots(df, x, y, z="", bins=4, ci=False, degree=3, norm_coefs=False, incl_intercept=True):

    df = df.replace([np_inf, -np_inf], np_nan).dropna()
    z_bin = z + '_bin'
    df[z_bin] = cut(df[z], bins=bins)
    ub = list(df[z_bin].unique().dropna())
    # print(ub)
    ub.sort()
    reg_res = []
    reg_res_y = []
    reg_res_p = []
    r_squared_results = []
    formulas = []

    for i in range(len(ub)):
        df_tmp = df[df[z_bin] == ub[i]]

        if df_tmp.shape[0] >= 5:

            X = df_tmp[x].values  # X value
            Y = df_tmp[y].values  # y value

            poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)  # Polynomial Transformation
            if norm_coefs:
                lr_model = Ridge(fit_intercept=incl_intercept)  # Linear Regression Model
            else:
                lr_model = LinearRegression(fit_intercept=incl_intercept)  # Linear Regression Model

            poly_features = poly.fit_transform(X.reshape(-1, 1))  # reshape X to polynomial
            x_range = linspace(quantile(X, 0.01), quantile(X, 0.98),
                               100)  # create a range of 100 point for plot, instead of min/max keep percentiles of X
            x_range_tr = poly.transform(x_range.reshape(-1, 1))  # transform plot x

            lr_model.fit(poly_features, Y)  # train model with reshaped X
            y_pred = lr_model.predict(x_range_tr)  # create presiction for plot X

            r_squared = lr_model.score(poly_features, Y)
            r_squared_results.append("For **" + z + "** value range of **" + str(ub[i]) + "**: ***~" + str(int(round(r_squared, 3) * 100)) + "% of the variance of " + y + " is explained by " + x + "*** in the Polynomlial Linear Regression model of degree "+str(3)+".")

            reg_res.extend(x_range)
            reg_res_y.extend(y_pred)
            reg_res_p.extend([str(ub[i])] * len(x_range))

            # Get Regression Formula
            formula = get_regression_formula(poly_reg_model=lr_model, degree=degree)
            formulas.append(str(ub[i]) + ' \\rightarrow ' + formula)

    df_reg = DataFrame(reg_res, columns=[x])
    df_reg[y] = reg_res_y
    df_reg[z_bin] = reg_res_p

    # fig = scatter(df_reg, x='x', y='y', facet_col=z_bin, marginal_x="box")
    fig = line(df_reg, x=x, y=y, facet_col=z_bin)

    fig.update_layout(
        title= x + " and " + y + " relationship plot for each " + z + " bin.",
        xaxis_title=x,
        yaxis_title=y,
        legend_title= z + " bin",
    )
    del df[z_bin]

    return fig, r_squared_results, formulas

# - - - - - - - - - - - - - - - -


def print_explainability_results(r_squared, model='Linear Regression', x='', y=''):
    with expander('The R-Squared metric'):
        markdown(
            "**Note**: To assess the goodness of fit for the trained " + model + " model, the **R-squared** (coefficient of determination) is indicated below, "
            "The R-squared value measures the proportion of the variance in the **dependent variable [" + y + "]** that can be explained by the **independent variable [" + x + "]**. "
            "of the variability in the dependent variable can be explained by the independent variable in the " + model + " model.")
    markdown("The R-Squared for this Regression is **" + str(round(r_squared, 4)) + "**, hence:")
    info("***~" + str(int(round(r_squared, 3) * 100)) + "% of the variance of " + y + " is explained by " + x + "***")
    return 0

@fragment
def get_regression_analysis_form():
    with form("dep_form"):
        use_z_var = selectbox('Type of Regression Analysis', options=[
            '[Fast] Bivariate (X, Y axes) with Linear Regression',
            '[Fast] 3-variate (X, Y, Z axes) with Linear Regression (Subplots)',
            '[Fast] 3-variate (X, Y, Z axes) with Linear Regression (Single)',
            '[Fast] Bivariate (X, Y axes) with KNN',
            '[Fast] 3-variate (X, Y, Z axes) with KNN (Single)',
            '[Fast] 3-variate (X, Y, Z axes) with KNN (Subplots)',
            '[Detailed] Bivariate (X, Y axes)',
            '[Detailed] 3-variate (X, Y, Z axes) (Subplots)',
            '[Detailed] 3-variate (X, Y, Z axes) (Single)'], index=0)
        dp_options = list(session_state.df._get_numeric_data().columns)
        cold = columns(2)
        with cold[0]:
            dep_x = selectbox(label='Select X Axis of Dependence Plot', options=dp_options, index=2,
                                 help="X-axis")
        with cold[1]:
            dep_y = selectbox(label='Select Y Axis of Dependence Plot', options=dp_options, index=3,
                                 help="Y-axis")

        with expander(label="Select Z Axis of Dependence Plot", expanded=False):
            colz = columns(2)
            with colz[0]:
                write('The *3-variate (X-Y-Z axes)* analysis will train a ***KNN*** ML model. KNN regression is '
                         'a non-parametric method that, in an intuitive manner, approximates the association between '
                         'independent variables and the continuous outcome by averaging the observations in the same neighbourhood.')
                write('The *Detailed 3-variate (X-Y-Z axes) Subplots* and *Detailed 3-variate (X-Y-Z axes) '
                         'Single Plot* train a ***Polynomial Linear Regression Model*** of a ***3rd degree polynomial*** '
                         'in order to fit X to Y. The subplots option will create a plot for each Z-bin while the '
                         'single plot will show everything in the same plot with different colors for each Z-Bin. '
                         'Along with the fitted line both detailed plots will provide a *Confidence Interval* for the regression line, along with a biined scatter plot of the actual Data')
                info('Because of the nature of the Models, the KNN Model is substantially faster.')

            with colz[1]:
                dep_z = selectbox(label='Select Z Axis of Dependence Plot', options=dp_options, index=4,
                                     help="Z-axis will be binned")
                z_binning_type = radio('Type of Binning', options=['(1) Automatic Binning', '(2) Custom Binning'],
                                          horizontal=True, index=0)
                dep_z_bins = slider(label='(1) Select Number of Bins for the Z-axis Feature', min_value=2,
                                       max_value=8, value=4,
                                       help="The bins will include approximately the same number of Data Points")
                write('(2) Define Custom Bins')
                dep_z_bins_custom = DataFrame([{"Bin Threshold": 0.0}])
                edited_df = data_editor(dep_z_bins_custom, num_rows="dynamic", use_container_width=True)
            code(
                'The Bins will be automatically generated based on the Chosen Number of Bins of the chosen (Z-axis) Feature, in order to have equally spaced interals, calculated based on the Data',
                language=None)
        with expander(label="Model & Plot Parameters", expanded=True):
            colp = columns(2)
            with colp[0]:
                norm_coefs = toggle('Normalize Coefficients (Ridge Regression)', value=False)
                show_scatter = toggle('Show Scattered Data', value=True)

                # with colp[2]:
                incl_intercept = toggle('Include Intercept', value=True)
            with colp[1]:
                reg_deg = slider(label='(1) Polynomial Degree of Regression Model', min_value=1,
                                    max_value=5, value=2,
                                    help="Degree Choice")

        code("🕑 Estimated Execution Time " + str(int(0.0002 * session_state.df.shape[0])) + " seconds, for " + str(
            session_state.df.shape[0]) + " samples.",
                language=None)
        submitted_dep = form_submit_button("Generate Dependence Plot")

        if submitted_dep:

            if z_binning_type == "(1) Automatic Binning":
                final_bins = dep_z_bins
            else:  # (2) Custom Binning
                final_bins = edited_df['Bin Threshold'].to_list()  # list(edited_df['Bin Threshold'].values)
                if len(final_bins) > 1:
                    if final_bins[-1] < session_state.df[dep_z].max():
                        final_bins.append(session_state.df[dep_z].max() + 0.01)

            # print(list(edited_df['Bin Threshold'].values))
            with spinner('Generating Dependence Plot for'):
                if use_z_var == '[Fast] Bivariate (X, Y axes) with Linear Regression':
                    if session_state.df.shape[0] < 300:
                        warning('💡 The DataTable size is too small in order to create a robust Linear Regression Model. Better use KNN.')
                    markdown(f"""This Plot examines the relationship between ***""" + dep_x + """*** and ***"""
                                + dep_y + """*** using the ***Polynomial Linear Regression Model of 3rd degree***.""")
                    r_fig, r_squared, formula = dependence_plot_plotly_lr(df=session_state.df[[dep_x, dep_y]], x=dep_x,
                                                                          y=dep_y, ci=False, degree=reg_deg, norm_coefs=norm_coefs, incl_intercept=incl_intercept, show_scatter=show_scatter)
                    print_explainability_results(r_squared, 'Polynomial Linear Regression', dep_x, dep_y)
                    write("Regression Formula for ***x  → " + dep_x + "*** and ***y  → " + dep_y + "***")
                    latex(formula)
                    plotly_chart(r_fig, use_container_width=True)
                    # SHOW FORMULA
                elif use_z_var == '[Fast] 3-variate (X, Y, Z axes) with Linear Regression (Subplots)':
                    if session_state.df.shape[0] < 1000:
                        warning('💡 The DataTable size is too small in order to create a robust Linear Regression Model. Better use KNN.')
                    markdown(f"""This Plot examines the relationship between ***""" + dep_x + """*** and ***"""
                                + dep_y + """*** for each of the ***""" + str(
                        final_bins) + """ """ + dep_z + """ bins***, using the ***Polynomial Linear Regression Model of 3rd degree***. """)
                    lr_fig, r_squared, formulas = dependence_plot_plotly_lr_3d_subplots(df=session_state.df[[dep_x, dep_y, dep_z]],
                                                                                        x=dep_x, y=dep_y, z=dep_z, bins=final_bins, ci=False, degree=reg_deg,
                                                                                        norm_coefs=norm_coefs, incl_intercept=incl_intercept)
                    for i in range(len(r_squared)):
                        info(r_squared[i])
                    write("Regression Formulas for ***x  → " + dep_x + "*** and ***y  → " + dep_y + "***")
                    for formula in formulas:
                        latex(formula)
                    plotly_chart(lr_fig, use_container_width=True)
                elif use_z_var == '[Fast] 3-variate (X, Y, Z axes) with Linear Regression (Single)':
                    if session_state.df.shape[0] < 1000:
                        warning('💡 The DataTable size is too small in order to create a robust Linear Regression Model. Better use KNN.')
                    markdown(f"""This Plot examines the relationship between ***""" + dep_x + """*** and ***"""
                                + dep_y + """*** for each of the ***""" + str(
                        final_bins) + """ """ + dep_z + """ bins***, using the ***Polynomial Linear Regression Model of 3rd degree***. """)
                    lr_fig, r_squared, formulas = dependence_plot_plotly_lr_3d(df=session_state.df[[dep_x, dep_y, dep_z]],
                                                                               x=dep_x, y=dep_y, z=dep_z, bins=final_bins, ci=False, degree=reg_deg,
                                                                               norm_coefs=norm_coefs, incl_intercept=incl_intercept)
                    for i in range(len(r_squared)):
                        info(r_squared[i])
                    write("Regression Formulas for ***x  → " + dep_x + "*** and ***y  → " + dep_y + "***")
                    for formula in formulas:
                        latex(formula)
                    plotly_chart(lr_fig, use_container_width=True)
                elif use_z_var == '[Fast] Bivariate (X, Y axes) with KNN':
                    markdown(f"""This Plot examines the relationship between ***""" + dep_x + """*** and ***"""
                                + dep_y + """*** using the ***KNN ML Model***. This Model finds the nearest
                                neighbors of each point and estimates their values. The value of each point is
                                therefore averaged and assigned to each point of the y axis""")
                    knn_fig, r_squared = dependence_plot_plotly_knn(x=dep_x, y=dep_y)
                    print_explainability_results(r_squared, 'KNN Regression', dep_x, dep_y)
                    plotly_chart(knn_fig, use_container_width=True)
                elif use_z_var == '[Fast] 3-variate (X, Y, Z axes) with KNN (Single)':
                    markdown(f"""This Plot examines the relationship between ***""" + dep_x + """*** and ***"""
                                + dep_y + """*** for each of the ***""" + str(final_bins) + """ """ + dep_z + """ bins***, using the ***KNN ML Model***. This Model finds the nearest
                                neighbors of each point and estimates their values. The value of each point is
                                therefore averaged and assigned to each point of the y axis""")
                    knn_fig, r_squared = dependence_plot_plotly_knn_3d(x=dep_x, y=dep_y, z=dep_z, bins=final_bins,
                                                                       type='single')
                    for i in range(len(r_squared)):
                        info(r_squared[i])
                    plotly_chart(knn_fig, use_container_width=True)
                elif use_z_var == '[Fast] 3-variate (X, Y, Z axes) with KNN (Subplots)':
                    markdown(f"""This Plot examines the relationship between ***""" + dep_x + """*** and ***"""
                                + dep_y + """*** for each of the ***""" + str(final_bins) + """ """ + dep_z + """ bins***, using the ***KNN ML Model***. This Model finds the nearest
                                neighbors of each point and estimates their values. The value of each point is
                                therefore averaged and assigned to each point of the y axis""")
                    knn_fig, r_squared = dependence_plot_plotly_knn_3d(x=dep_x, y=dep_y, z=dep_z, bins=final_bins)
                    for i in range(len(r_squared)):
                        info(r_squared[i])
                    plotly_chart(knn_fig, use_container_width=True)
                elif use_z_var == "[Detailed] Bivariate (X, Y axes)":
                    dep_plot = dependence_plot(x=dep_x, y=dep_y, degree=reg_deg)
                    pyplot(dep_plot)
                    # dep_plot.savefig("../cache/dependece_plot_" + x + "_" + y+".png")
                elif use_z_var == "[Detailed] 3-variate (X, Y, Z axes) (Subplots)":
                    pyplot(dependence_plot_3d_subplots(x=dep_x, y=dep_y, z=dep_z, bins=final_bins, degree=reg_deg))
                else:  # Detailed 3-variate (X-Y-Z axes) Single Plot
                    dep_plot, bin_seq, c_palette = dependence_plot_3d(x=dep_x, y=dep_y, z=dep_z, bins=final_bins, degree=reg_deg)
                    markdown(f"""Bins for {dep_z}:""")
                    bin_cols = columns(len(bin_seq))
                    for i in range(len(bin_seq)):
                        with bin_cols[i]:
                            markdown(
                                "<p style='color: " + c_palette[i] + ";margin: 0;'> ● <span style='color: grey;'>" +
                                bin_seq[i] + "</span> </p>", unsafe_allow_html=True)
                    # fig_html = fig_to_html(dep_plot, d3_url="http://0.0.0.0:8000/d3.v5.min.js", mpld3_url="http://0.0.0.0:8000/mpld3.v0.5.9.min.js", include_libraries=True)
                    fig_html = fig_to_html(dep_plot)
                    components_html(fig_html, height=530)
                    code("""⬆️⬆️ You can control the plot using these controls""", language=None)
