from streamlit import pyplot, plotly_chart, write, warning
from plotly.graph_objects import Figure, Scatter
from statsmodels.api import qqplot
from scipy.stats import linregress
import matplotlib.pyplot as plt
from numpy import linspace


def residual_vs_predicted_plot(y_actual, y_predicted, title="Residual vs. Predicted"):
    residuals = y_actual - y_predicted
    ### Plotly ###
    # fig = scatter(x=y_predicted, y=residuals, labels={'x': 'Predicted', 'y': 'Residuals'}, trendline="ols",
    # trendline_options=dict(log_x=True), trendline_color_override="red")  #  title=title,
    # fig.update_layout(yaxis_title="Residuals")
    # plotly_chart(fig, use_container_width=True)
    print(len(y_predicted), len(residuals))
    try:
        slope, intercept, r_value, p_value, std_err = linregress(y_predicted, residuals)
    except ValueError:
        slope, intercept, r_value = 0, 0, 0
        pass

    fig = plt.figure(figsize=(16, 12))
    plt.scatter(y_predicted, residuals, alpha=0.7)
    reg_line_x = linspace(min(y_predicted), max(y_predicted), 100)
    reg_line_y = slope * reg_line_x + intercept
    plt.plot(reg_line_x, reg_line_y, color='red', linestyle='--',
             label=f'Regression Line (R-squared={r_value ** 2:.2f})')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    # plt.title(title)
    plt.legend()
    plt.grid(True)
    pyplot(fig)
    return


def error_bars_plot(y_actual, y_predicted, confidence_interval, title="Error Bars"):
    fig = Figure()
    fig.add_trace(Scatter(x=y_actual, y=y_predicted, mode='markers', name='Predicted'))
    fig.add_trace(Scatter(x=y_actual, y=y_actual + confidence_interval, mode='lines', name='Upper Bound'))
    fig.add_trace(Scatter(x=y_actual, y=y_actual - confidence_interval, mode='lines', name='Lower Bound'))
    fig.update_layout(xaxis_title="Actual", yaxis_title="Predicted", title=title)
    plotly_chart(fig, use_container_width=True)


def qq_plot(y_actual, y_predicted, title="QQ Plot"):
    residuals = y_actual - y_predicted
    qqplot(residuals, line='45', fit=True)

    fig = qqplot(residuals, line='45', fit=True, loc=4)
    fig.figsize = (8, 6)
    # fig.update_layout(title=title)
    pyplot(fig, use_container_width=True)#, use_container_width=True)
    return


def residual_histogram(y_actual, y_predicted, title="Histogram of Residuals"):
    residuals = y_actual - y_predicted
    # fig = create_distplot([residuals], group_labels=['Residuals'], colors=['blue'], bin_size=0.1, show_rug=False)
    # fig.update_layout(xaxis_title="Residuals", yaxis_title="Density")  # title=title

    fig = plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=30, color="blue", alpha=0.7, edgecolor="black")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.grid(True)
    pyplot(fig)  #,use_container_width=True)
    return


def compare_distributions(y_actual, y_predicted, title="Comparison of Distributions"):
    fig = plt.figure(figsize=(6, 4))
    plt.hist(y_actual, bins=30, color='blue', alpha=0.5, label='Actual', edgecolor='black')
    plt.hist(y_predicted, bins=30, color='red', alpha=0.5, label='Predicted', edgecolor='black')
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    # plt.title(title)
    plt.legend()
    plt.grid(True)
    pyplot(fig)


def actual_vs_predicted_plot(y_test, y_pred):
    x = []
    for i in range(len(y_test)):
        x.append(i)

    if len(x) > 10000:
        sampling_step = int(len(x) / 10000)
        warning('The Size of the Dataset is too **large** to show. The Plot will contain a **resampled** version.')
    else:
        sampling_step = 1

    fig = Figure()
    fig.add_trace(Scatter(x=x[1::sampling_step], y=y_test[1::sampling_step], name="Actual", marker=dict(color='blue')))
    fig.add_trace(Scatter(x=x[1::sampling_step], y=y_pred[1::sampling_step], name="Prediction", marker=dict(color='orange')))

    plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))
    return
