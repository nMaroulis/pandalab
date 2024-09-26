from streamlit import write, warning, success, markdown, divider, dataframe, columns, spinner, caption, expander
from library.modelling.evaluation.evaluation_plots import (qq_plot, residual_vs_predicted_plot, residual_histogram,
                                                         compare_distributions, actual_vs_predicted_plot)
from numpy import std as np_std, median as np_median
from pandas import DataFrame
from library.modelling.models.dnn.LSTM import LSTMModel


def show_indications(r2, mean_absolute_error, y_test):
    write("Automated Indications based on Model's MAE and R^2 Scores:")
    if r2 < 0:
        warning("💡 The Model's **R-squared** value is **EXTREMELY LOW**. The Model did not capture at all the variance of the Target Variable.")
    elif 0.0 <= r2 < 0.2:
        warning("💡 The Model's **R-squared** value is **VERY LOW**. The Model did not capture the variance of the Target Variable.")
    elif 0.2 <= r2 < 0.4:
        warning("💡 The Model's **R-squared** value is **LOW**. The Model did not capture the variance of the Target Variable.")
    elif 0.4 <= r2 < 0.6:
        warning("💡 The Model's **R-squared** value is **MODERATE**. The Model captured some of the variance of the Target Variable.")
    elif 0.6 <= r2 < 0.8:
        warning("💡 The Model's **R-squared** value is **GOOD**. There's room for Improvement.")
    elif 0.8 <= r2 < 0.9:
        success("💡 The Model's **R-squared** value is **VERY GOOD**.")
    elif 0.9 <= r2 <= 1.0:
        success("💡 The Model's **R-squared** value is **PERFECT**.")
    else:  # bad value
        pass

    if mean_absolute_error <= np_std(y_test):
        success("💡 The Model's **MAE** is **lower** than the **Standard Deviation** of the **Target variable**, which might indicate a relatively good Model Performance.")
    else:
        warning("💡 The Model's **MAE** is **higher** than the **Standard Deviation** of the **Target variable**, which might indicate a bad Model Fit on the Target.")
    if mean_absolute_error < np_median(y_test):
        success("💡 The Model's **MAE** is **lower** than the **Median** of the **Target variable**, suggesting that the prediction lies within a plausible range.")
    else:
        warning("💡 The Model's **MAE** is **larger** than the **Median** of the **Target variable**, which indicates that the Prediction nees improvement.")
    return


def show_model_scores(mean_absolute_error, rmse, max_error,  rae, r2, training_time, old_size, new_size, training_params_dict):

    # FIX TIME FORMAT FOR PRINT
    training_time_h, remainder = divmod(training_time, 3600)
    training_time_m, training_time_s = divmod(remainder, 60)
    training_time_h, training_time_m, training_time_s = str(int(training_time_h)), str(int(training_time_m)), str(round(training_time_s, 1))

    markdown("<h2 style='text-align: left;margin-top:0.5em;margin-bottom:0'>Model Evaluation</h2>", unsafe_allow_html=True)
    divider()
    markdown("<h3 style='text-align: left; color: #48494B;margin-top:0.5em;'>1. Model Error Scores</h3>", unsafe_allow_html=True)

    caption('Expand to see Training Parameters:')
    with expander('Training Parameters'):
        tc0, tc1, tc2 = columns(3)
        with tc0:
            write('Input Features', len(training_params_dict['input_features']))
            write(training_params_dict['input_features'])
            if training_params_dict['data_norm']:
                caption("with Data Normalization: " + str(training_params_dict['data_norm']))
        with tc1:
            write('Model: ' + training_params_dict['model'].replace("[Auto]", ""))
            write(training_params_dict['hyperparams'])
        with tc2:
            write('Output Feature')
            write(training_params_dict['target_variable'])

    write("> 🕒 The **training time** was " + training_time_h + " Hours " + training_time_m + " Minutes and " + training_time_s + " Seconds" )
    write('> 📝 The **Evaluation** is based on the **Test Set** of size **' + str(old_size) + ' samples**.')  # , where ' + str(old_size - new_size) + ' samples were not used because of Empty Values.')
    write("The ***Evaluation Metrics*** based on the trained Machine Learning Model's Predictions:")
    model_scores = [["Mean Absolute Error (MAE)", mean_absolute_error],
           ["Root Mean Squared Error (RMSE)", rmse],
           ["Coefficient of Determnation (R^2)", r2],
           ["Max Error", max_error],
           ["Relative Absolute Error (RAE)", rae]]
    model_scores = DataFrame(model_scores, columns=['Metric', 'Score'])
    dataframe(model_scores, use_container_width=True, hide_index=True)
    return


def show_def_residual_analysis(y_test, y_pred):
    markdown("<h3 style='text-align: left; color: #48494B;margin-top:2em;'>2. Residual Analysis</h3>",
             unsafe_allow_html=True)
    write(
        "Residuals are estimates of experimental error obtained by subtracting the observed responses (actual target) from the predicted responses (model's prediction/estimation), hence the differences between the actual target values and the corresponding predictions made by the regression model")
    markdown("<br>", unsafe_allow_html=True)
    eval_col0, eval_col1 = columns([2, 3])
    with eval_col0:
        markdown("<h5 style='text-align: center; color: #48494B;'>QQ Plot on Residuals</h5>", unsafe_allow_html=True)
        # with expander('Explanation'):
        markdown(
            """<p style='line-height: 1.1;'>A Quantile-Quantile (QQ) plot is a scatter plot designed to compare the data to the theoretical distributions to visually determine if the observations are likely to have come from a known population. The empirical quantiles are plotted to the y-axis, and the x-axis contains the values of the theorical model. A 45-degree reference line is also plotted. If the empirical data come from the population with the choosen distribution, the points should fall approximately along this reference line. The larger the departure from the reference line, the greater the evidence that the data set have come from a population with a different distribution.</p>""",
            unsafe_allow_html=True)
        with spinner('Generating Q-Q Plot...'):
            qq_plot(y_test, y_pred)
    with eval_col1:
        markdown("<h5 style='text-align: center; color: #48494B;'>Model Prediction (x-axis) vs Residual (y-axis)</h5>",
                 unsafe_allow_html=True)
        # with expander('Explanation'):
        markdown(
            """<p style='line-height: 1.1;'> The scatter points represent the residuals. Each point represents an individual data point in the test set. The red dashed line in the plot is a linear regression line fitted to the residuals. It helps visualize any trend or pattern in the residuals concerning the predicted values. <strong>Interpretation:</strong> If the scatter points are randomly distributed around the horizontal axis (at zero), it suggests that the residuals exhibit no systematic pattern with respect to predicted values. This is indicative of a well-behaved regression model. - Any noticeable curvature or consistent pattern in the residuals may suggest that the model has limitations or assumptions that are not met. In summary, the "Residual vs. Predicted" plot helps you assess the model's performance by visualizing how the residuals vary in relation to the predicted values. A well-behaved plot with a random distribution of points around zero indicates a good model fit, while patterns or trends may require further investigation.</p>""",
            unsafe_allow_html=True)
        with spinner('Generating Residual Plot...'):
            residual_vs_predicted_plot(y_test, y_pred)
    eval_col2, eval_col3 = columns(2)
    with eval_col2:
        markdown("<h5 style='text-align: center; color: #48494B;margin-top:0.5em;'>Residual Histogram</h5>",
                 unsafe_allow_html=True)
        markdown(
            """<p style='line-height: 1.1;'>The Histogram of the Residual can be used to check whether the variance is normally distributed. A symmetric bell-shaped histogram which is evenly distributed around zero indicates that the normality assumption is likely to be true. If the histogram indicates that random error is not normally distributed, it suggests that the model's underlying assumptions may have been violated.</p>""",
            unsafe_allow_html=True)
        with spinner('Generating Residual Histogram...'):
            residual_histogram(y_test, y_pred)
    with eval_col3:
        markdown(
            "<h5 style='text-align: center; color: #48494B;margin-top:0.5em;'>Actual Data vs Predicted Distributions</h5>",
            unsafe_allow_html=True)
        markdown(
            """<p style='line-height: 1.1;'>This plot visually compares the distributions of actual (blue) and predicted (red) valuesfrom the test set. By comparing these two distributions in a single plot, we assess how closely the predicted values align with the actual values. A good model should have predicted values that closely match the distribution of the actual values. Discrepancies or differences between the two histograms can provide insights into the model's performance and areas where it may need improvement.</p>""",
            unsafe_allow_html=True)
        with spinner('Generating Distributions...'):
            compare_distributions(y_test, y_pred, title="Comparison of Distributions")
    return


def show_prediction_analysis(y_test, y_pred):

    markdown("<h3 style='text-align: left; color: #48494B;margin-top:2em;'>3. Prediction Visualization</h3>", unsafe_allow_html=True)
    markdown("<h5 style='text-align: center; color: #48494B;'>Model Prediction (orange) vs Actual Value (blue)</h5>",
             unsafe_allow_html=True)
    actual_vs_predicted_plot(y_test, y_pred)

    return
