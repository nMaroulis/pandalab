import shap
import streamlit.components.v1 as components
import xgboost
from re import compile, IGNORECASE
from sklearn.model_selection import train_test_split
import numpy as np
from streamlit import session_state, set_option, pyplot as st_pyplot, write, header, subheader, spinner, tabs
from matplotlib.pyplot import subplots as plt_subplots, figure as plt_figure
from pandas import DataFrame
from sklearn.inspection import permutation_importance

#---------------- SHAP ------------------------

def get_automatic_model_speed_dict(selected_ams):

    ams_dict = {
        'Express [30s]': 100000,
        'Fast [1.5m]': 300000,
        'Normal [3m]': 550000,
        'Slow [6m]': 900000,
        'All Data [~12m]': 100000000
    }
    return ams_dict.get(selected_ams)


def remove_collinear(x):
    correlations = x.corr(method='pearson', numeric_only=True).abs()
    upper_tr = correlations.where(np.triu(np.ones(correlations.shape), k=1).astype('bool'))
    to_drop = [column for column in upper_tr.columns if any(upper_tr[column] > 0.95)]
    return to_drop


def remove_null(x):
    null_cols_to_remove = list(x.loc[:, x.isnull().mean() > 0.4].columns)
    return null_cols_to_remove


def get_data(model_inputs, target, automatic_model_speed, use_train_set_for_all):

    all_cols = model_inputs + [target]

    # SAMPLE IF TOO LARGE SIZE
    if session_state.df[all_cols].dropna(axis=0).shape[0] > get_automatic_model_speed_dict(automatic_model_speed):
        sample_size = get_automatic_model_speed_dict(automatic_model_speed)
    else:
        # sample_size = session_state.df[all_cols].dropna(axis=0).shape[0]
        sample_size = session_state.df[all_cols].shape[0]

    df = session_state.df[all_cols].sample(sample_size).copy()

    null_columns = remove_null(df)
    df = df.drop(null_columns, axis=1)

    for c in null_columns:
        if c in model_inputs:
            model_inputs.remove(c)

    df = df.dropna(axis=0)  # .dropna(axis=0)
    X = df[model_inputs]

    autocorrelated_columns = remove_collinear(X)
    X = X.drop(autocorrelated_columns, axis=1)


    regex = compile(r"\[|\]|<", IGNORECASE)
    X.columns = [regex.sub("", col) if any(i in str(col) for i in set(('[', ']', '<'))) else col for col in X.columns.values]
    y = df[target]

    if use_train_set_for_all:
        x_train = X
        x_test = X
        y_train = y
        # y_test = # train_test_split(X, y, train_size=1, random_state=42)
        y_test = y
    else:
        x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    return x_train, x_test, y_train, y_test, autocorrelated_columns, null_columns


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


def get_xgboost_shap_analysis(model_inputs, target, automatic_model_speed='Normal', shap_plots=(0, 0, 0, 0, 0),
                              estimators=300, learning_rate=0.01, max_depth=8, min_child_weight=18, gamma=0,
                              subsample=1.0, cluster_num='None', use_train_set_for_all=False):

    header("Feature Importance using SHAP Values")
    # TRAIN TEST SPLIT
    X_train, X_test, y_train, y_test, autocorrelated_columns, null_columns = get_data(model_inputs, target, automatic_model_speed, use_train_set_for_all)

    # TRAIN XGBOOST MODEL
    with spinner('Training XGBoost ML Model...'):
        model = xgboost.train({"learning_rate": learning_rate, "subsample": subsample, 'n_jobs': 16}, xgboost.DMatrix(X_train, label=y_train), estimators)
        print('LOG :: Feature Importance - SHAP :: XGBoost Model trained')

    # GET RESULT AND PRINT
    with spinner('Generating Results...'):

        y_pred = model.predict(xgboost.DMatrix(X_test))
        score = np.mean(np.abs(y_test - y_pred))
        subheader('XGBoost ML Model training results:')
        write("""💡 The following Features were removed from the Input space, as a very High Linear correlation (Pearson Correlation ρ >0.95) was found with another feature in the input space (***Multi-Colinearity***). See Pearson Correlation Heatmap on the Correlation Tab for more details.""")
        write(autocorrelated_columns)
        write("""💡 The following Features were removed from the Input space, as they contain more than 40% of Empty Values.""")
        write(null_columns)
        write('💡 XGBoost Model approximated ' + target + ' with a Mean Absolute Error (MAE) of ' + str(round(score, 3)))

    subheader('SHAP Values')
    with spinner('Calculating SHAP values...'):

        shap.initjs()
        # Explain the model's predictions using SHAP
        explainer = shap.Explainer(model)  # Fit Explainer, other option is explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test)
        # set_option('deprecation.showPyplotGlobalUse', False)  # no warning from pyplot shap
        t1, t2, t3, t4, t5 = tabs(['Global Feature Importance', 'Summary Plot', 'Scatter Plot', '3-Variate Scatter Plot', 'Heatmap'])

        with t1:
            if shap_plots[0]:
                if cluster_num == 'None':
                    write("**SHAP Feature Importance (absolute & mean)**")
                    write("The goal of SHAP is to explain the prediction of an instance x by computing the contribution of each "
                          "feature to the prediction. The SHAP explanation method computes Shapley values from coalitional game "
                          "theory. The feature values of a data instance act as players in a coalition. Shapley values tell us "
                          "how to fairly distribute the prediction) among the features.")
                    fi_shap = shap_values.abs.sum(0)
                    fi_shap = fi_shap / fi_shap.sum()
                    # fig = shap.plots.bar(fi_shap, max_display=50)

                    # fig = shap.dependence_plot(1, shap_values.values, X_test)

                    fig, ax = plt_subplots(nrows=1, ncols=1, figsize=(20, len(model_inputs)))
                    shap.plots.bar(fi_shap, max_display=50, ax=ax, show=False)
                    st_pyplot(fig)
                else:
                    write("SHAP Feature Importance for " + cluster_num + " Clusters.")
                    fig, ax = plt_subplots(nrows=1, ncols=1, figsize=(20, len(model_inputs)))
                    shap.plots.bar(shap_values.cohorts(int(cluster_num)).abs.mean(0), max_display=50, ax=ax, show=False)
                    st_pyplot(fig)
            else:
                write('Option not chosen in SHAP Parameters form')
        with t2:
            if shap_plots[1]:
                write("SHAP Feature Importance Summary")
                fig, ax = plt_subplots(nrows=1, ncols=1, figsize=(20, len(model_inputs)))
                shap.plots.beeswarm(shap_values, max_display=20, show=False)
                st_pyplot(fig)
            else:
                write('Option not chosen in SHAP Parameters form')
        with t3:
            if shap_plots[2]:
                write("**SHAP Feature Importance Scatter Plot for each Input**")
                for i in range(0, len(X_test.columns), 2):
                    fig, (ax1, ax2) = plt_subplots(nrows=1, ncols=2, figsize=(20, 5))
                    shap.plots.scatter(shap_values[:, X_test.columns[i]], ax=ax1, show=False)
                    if i+1 < len(X_test.columns):
                        shap.plots.scatter(shap_values[:, X_test.columns[i+1]], ax=ax2, show=False)
                    st_pyplot(fig, use_container_width=True)
            else:
                write('Option not chosen in SHAP Parameters form')
        with t4:
            if shap_plots[3]:
                write("**SHAP Feature Importance 3-Variate Scatter Plot for each Input**")
                for i in range(0, len(X_test.columns), 2):
                    fig, (ax1, ax2) = plt_subplots(nrows=1, ncols=2, figsize=(20, 5))
                    shap.dependence_plot(i, shap_values.values, X_test, ax=ax1, show=False)
                    # ax1.set_xlabel('', fontsize=6)
                    if i+1 < len(X_test.columns):
                        shap.dependence_plot(i+1, shap_values.values, X_test, ax=ax2, show=False)
                    st_pyplot(fig, use_container_width=True)
            else:
                write('Option not chosen in SHAP Parameters form')
        with t5:
            if shap_plots[4]:
                write("SHAP Feature Importance ***Heatmap***.")
                fig, ax = plt_subplots(1,1,figsize=(24, 24))
                shap.plots.heatmap(shap_values, max_display=16, ax=ax, show=False)
                st_pyplot(fig, use_container_width=True)
            else:
                write('Option not chosen in SHAP Parameters form')

        # # st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:])) # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
        # st_shap(shap.force_plot(explainer.expected_value, shap_values, X_test), 400)  # visualize the training set predictions


        # OTHER OPTIONS
        # pyplot(shap.plots.heatmap(shap_values[:1000])) # after taking .max() of values
        # pyplot(shap.plots.waterfall(shap_values[1], max_display=20))
    return 1


# -------------- XGBoost Feature Importance ------------------------
def get_default_feature_importance(df, estimators=300, learning_rate=0.015, max_depth=8, min_child_weight=18, gamma=0, subsample=1.0, target="", model_inputs=None):
    # XGBoost Hyperparameters - Optuna Optimized Parameters
    params = {'lambda': 1.7, 'alpha': 9.7, 'colsample_bytree': 0.9, 'subsample': subsample, 'learning_rate': learning_rate,
              'n_estimators': estimators, 'min_split_loss': gamma,
              'max_depth': max_depth, 'random_state': 2020, 'min_child_weight': min_child_weight, 'tree_method': 'auto'}

    xgb_model = xgboost.XGBRegressor(**params)

    x = df[model_inputs].dropna(axis=0)
    y = df[target].dropna(axis=0)

    regex = compile(r"\[|\]|<", IGNORECASE)
    x.columns = [regex.sub("", col) if any(i in str(col) for i in set(('[', ']', '<'))) else col for col in x.columns.values]
    # y.columns = [regex.sub("_", col) if any(i in str(col) for i in set(('[', ']', '<'))) else col for col in y.columns.values]

    print('LOG :: models :: Training XGBoost Model for', df.shape[0], 'training samples.')
    xgb_model.fit(x, y)
    print('LOG :: models :: Training Completed Successfully and Saved to Memory')

    fig, ax = plt_subplots(1, 1, figsize=(20, 10))
    # xgb_model.get_booster().feature_names = list(x.columns)  # c_cols df.drop(cols_to_drop, axis=1)
    # plot_importance(xgb_model.get_booster(), ax=ax, height=0.6, title='Feature importance (Weight)')

    sorted_idx = xgb_model.feature_importances_.argsort()
    # print(xgb_model.feature_importances_)
    ax.barh(x.columns[sorted_idx], xgb_model.feature_importances_[sorted_idx])
    ax.set_xlabel("Feature Importance for the ML Model [0,1]")
    ax.grid(axis='y')
    # plot_importance(xgb_model.get_booster(), ax=ax2, height=0.3, importance_type='gain',
    #                     title='Feature importance (Gain)')
    #
    # plot_importance(xgb_model.get_booster(), ax=ax3, height=0.3, importance_type='cover',
    #                     title='Feature importance (Cover)')

    return fig

### ----------------------- XGBoost Permutation Importance --------------------------

def get_xgboost_permutation_importance(df, estimators=300, learning_rate=0.015, max_depth=8, min_child_weight=18, gamma=0, subsample=1.0, target: str ="", model_inputs=None):
    # XGBoost Hyperparameters - Optuna Optimized Parameters
    params = {'lambda': 1.7, 'alpha': 9.7, 'colsample_bytree': 0.9, 'subsample': subsample, 'learning_rate': learning_rate,
              'n_estimators': estimators, 'min_split_loss': gamma,
              'max_depth': max_depth, 'random_state': 2020, 'min_child_weight': min_child_weight, 'tree_method': 'auto'}

    xgb_model = xgboost.XGBRegressor(**params)

    if model_inputs is None:
        x = df.dropna(axis=0).drop([target], axis=1)
        x = x.select_dtypes(['number'])
        y = df.dropna(axis=0)[target]
    else: # user specified inputs
        x = df.dropna(axis=0)[model_inputs]
        y = df.dropna(axis=0)[target]

    regex = compile(r"\[|\]|<", IGNORECASE)
    x.columns = [regex.sub("", col) if any(i in str(col) for i in set(('[', ']', '<'))) else col for col in x.columns.values]


    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42)
    xgb_model.fit(X_train, y_train)
    result = permutation_importance(
        xgb_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)

    sorted_importances_idx = result.importances_mean.argsort()
    importances = DataFrame(
        result.importances[sorted_importances_idx].T,
        columns=x.columns[sorted_importances_idx],
    )

    ax = importances.plot.box(vert=False, whis=10)
    ax.set_title("Permutation Importances (test set)")
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel("Decrease in accuracy score")
    ax.figure.tight_layout()
    # ax.figure.figsize = (38,34)
    ax.title.set_fontsize(6)
    ax.xaxis.label.set_size(6)
    ax.yaxis.label.set_size(6)
    ax.tick_params(axis='x', labelsize=4)
    ax.tick_params(axis='y', labelsize=4)

    print('LOG :: models :: Permutation :: Training Completed Successfully and Saved to Memory')

    return ax.figure
