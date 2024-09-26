from streamlit import session_state, cache_data, plotly_chart, write, columns, spinner
# import matplotlib.pyplot as plt
from math import ceil
# from seaborn import heatmap, kdeplot, boxplot, violinplot
from numpy import number
from pandas import DataFrame, Timedelta
# from settings.feature_map import feature_selection
from plotly.graph_objects import Histogram, Figure, Box, Scatter
from plotly.subplots import make_subplots
from plotly.express import pie


def get_feature_status_pie_chart():

    sizes = [session_state.df.shape[1], session_state['cleaned features'], session_state['filtered features'][0],
             session_state['filtered features'][1], session_state['filtered features'][2],  session_state['filtered features'][3],
             session_state['low corr features'], session_state['generated features']]
    labels = ["Current Features", "Data Cleaning", "Static Filter", "Colinearity Filter", 'Manual Removal Filter', 'Categorical Features', "Low Correlation", "Generated Features"]
    plt_sizes = []
    plt_labels = []
    for s in range(len(sizes)):
        if sizes[s] > 0:
            plt_sizes.append(sizes[s])
            plt_labels.append(labels[s])

    # PLOTLY WAY
    fig = pie(labels, values=plt_sizes, hole=0.6,
              names=plt_labels, color=plt_labels,
              title='DataTable Features',
              color_discrete_map={'Current Features': '#50C878',
                                  'Data Cleaning': '#E78587',
                                  'Static Filter': '#6082B6',
                                  'Colinearity Filter': '#6495ED',
                                  'Manual Removal Filter': '#6061b6',
                                  'Categorical Features': '#60b69f',
                                  'Low Correlation': '#FFD580',
                                  'Generated Features': '#388c54',
                                  })
    fig.update_traces(
        # title_font=dict(size=16, family='Verdana',
        #                 color='darkred'),
        hoverinfo='label+value',
        textinfo='value+percent')
    plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
    return fig

@cache_data(show_spinner="Generating Null Pie Chart...")
def get_feature_null_pie_chart():

    null_vals = session_state.df.isna().sum().sum()
    total_size = session_state.df.shape[0] * session_state.df.shape[1]
    sizes = [total_size-null_vals, null_vals]
    labels = ["Valid Samples", "Null Samples"]
    print(null_vals, session_state.df.shape)
    # PLOTLY WAY
    fig = pie(labels, values=sizes, hole=0.6,
              names=labels, color=labels,
              title='DataTable Null Samples',
              color_discrete_map={'Valid Samples': '#0077B5',
                                  'Null Samples': '#E68523',
                                  })
    fig.update_traces(
        hoverinfo='label+value+percent',
        textinfo='value+percent')
    plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
    return fig



def df_density_plots_plotly(df=None):

    fig = make_subplots(rows=ceil((df.shape[1]) / 5), cols=5)
    x, y = 1, 1
    for i in df.select_dtypes(include=number).columns.tolist():

        cl = '#0047AB'
        fig.add_trace(Histogram(x=df[i], name=i, autobinx=True, histnorm='percent', marker_color=cl, hoverinfo="skip"), x, y)
        # fig.add_trace(Violin(x=df[i], name=i, box_visible=True, meanline_visible=True, hoverinfo="skip"), x, y)

        y += 1
        if y > 5:
            y = 1
            x += 1
        # fig.update_layout(title_text='Curve and Rug Plot')
    fig.update_layout(width=400, height=1200)
    return fig


def get_feature_descriptive_analytics(column_selected_info):
    if column_selected_info != '<None>':

        with spinner('Generating Analysis for ' + column_selected_info):

            selected_feature = session_state.df[column_selected_info].copy()
            # Create a Subplot with 3 columns
            fig = make_subplots(rows=1, cols=3, subplot_titles=(f"Histogram for {column_selected_info}",
                                                                f"Boxplot for {column_selected_info}",
                                                                f"Percentile Plot for {column_selected_info}"))
            # Add Histogram
            fig.add_trace(Histogram(x=selected_feature, name="Histogram"), row=1, col=1)
            # Add Boxplot
            fig.add_trace(Box(y=selected_feature, name="Boxplot"), row=1, col=2)
            # Add Q-Q Plot
            # Add Percentile Plot
            percentile_values = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            percentiles = [selected_feature.quantile(0), selected_feature.quantile(0.1), selected_feature.quantile(0.2),
                           selected_feature.quantile(0.3), selected_feature.quantile(0.4), selected_feature.quantile(0.5),
                           selected_feature.quantile(0.6), selected_feature.quantile(0.7), selected_feature.quantile(0.8),
                           selected_feature.quantile(0.9), selected_feature.quantile(1)
                           ]  # [np_perc(selected_feature, p) for p in [0,25,50,75,100]]
            fig.add_trace(Scatter(x=percentile_values, y=percentiles, mode="lines", name="Percentile Plot"),
                          row=1, col=3)
            # Update layout
            fig.update_layout(title=f"Statistics and Plots for {column_selected_info}",
                              showlegend=False,
                              height=400)

            # Show Plotly figure
            plotly_chart(fig, use_container_width=True)
        with spinner('Generating Analysis for ' + column_selected_info):

            f_col0, f_col1 = columns([1, 3])
            with f_col0:
                write(session_state.df[column_selected_info].describe())
            with f_col1:
                subsample_step = 1
                if selected_feature.shape[0] > 2000:
                    subsample_step = int(selected_feature.shape[0] / 2000)
                if "DateTime" in session_state.df.columns:
                    fig = Figure(data=Scatter(x=session_state.df.iloc[::subsample_step, :]["DateTime"], y=session_state.df.iloc[::subsample_step, :][column_selected_info]))
                else:
                    fig = Figure(data=Scatter(x=session_state.df.iloc[::subsample_step, :].index, y=session_state.df.iloc[::subsample_step, :][column_selected_info]))
                fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
                # fig.update_xaxes(tickangle=45)
                plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)


@cache_data
def get_missing_values():

    missing_values = (session_state.df.isnull().sum() / session_state.df.shape[0]) * 100
    non_missing_values = (session_state.df.notnull().sum() / session_state.df.shape[0]) * 100

    chart_data = []
    c = 0
    for i in session_state.df.columns.to_list():
        chart_data.append([i,  non_missing_values[c], missing_values[c], 0])
        c += 1
    chart_data = DataFrame(chart_data, columns=['Name', '1. Valid Features [%]', '2. Invalid Features [%]', '3. Missing'])
    chart_data.set_index('Name', inplace=True)

    return chart_data


@cache_data
def get_percentiles_values(args):
    p0 = [session_state.df[args[0]].mean(), session_state.df[args[0]].std(), session_state.df[args[0]].min(),
     session_state.df[args[0]].quantile(0.25), session_state.df[args[0]].quantile(0.5),
     session_state.df[args[0]].quantile(0.75), session_state.df[args[0]].quantile(0.95), session_state.df[args[0]].max()]

    if len(args) > 1:
        p1 = [session_state.df[args[1]].mean(), session_state.df[args[1]].std(), session_state.df[args[1]].min(),
              session_state.df[args[1]].quantile(0.25), session_state.df[args[1]].quantile(0.5),
              session_state.df[args[1]].quantile(0.75), session_state.df[args[1]].quantile(0.95), session_state.df[args[1]].max()]
    else:
        p1 = None
    return p0, p1
