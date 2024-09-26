from streamlit import session_state, form, form_submit_button, plotly_chart, code, spinner, multiselect, selectbox,\
    columns, radio, write, warning, info, error, container, dataframe, caption, markdown, cache_data
from plotly.graph_objects import Line
from plotly.subplots import make_subplots
from plotly.express import pie, histogram, box, violin
from numpy import number as np_number
from plotly.graph_objects import Scatter, Scattergl

### --------- LINE PLOT TOOL


def plotly_line_plot_many(df, plt_columns):

    fig = make_subplots(len(plt_columns), 1)

    for i in range(len(plt_columns)):

        fig.add_trace(Scatter(x=df.index, y=df[plt_columns[i]], hoverinfo='skip', name=plt_columns[i]), i+1, 1)
    fig.update_xaxes(matches='x')
    fig.update_xaxes(showticklabels=False)  # hide all the xticks
    fig.update_xaxes(showticklabels=True, row=len(plt_columns), col=1)
    fig.update_layout(height=1000)

    return fig


def plotly_line_plot(lp_y, lp_sampling, lp_sampling_type="First"):


    # Resampling options
    if lp_sampling == "1 minute": sampling_rate = "1min"
    elif lp_sampling == "5 minutes": sampling_rate = "5min"
    elif lp_sampling == "10 minutes":sampling_rate = "10min"
    elif lp_sampling == "30 minutes": sampling_rate = "30min"
    elif lp_sampling == "1 hour": sampling_rate = "60min"
    elif lp_sampling == "6 hours": sampling_rate = "300min"
    else: sampling_rate = None

    df_plot = session_state.df[lp_y].copy()
    try:
        if sampling_rate: # Resample based on sampling rate, otherwise keep original and set as index the datetime
            if lp_sampling_type == 'Mean':
                df_plot = df_plot.resample(sampling_rate).mean()
            if lp_sampling_type == 'Min':
                df_plot = df_plot.resample(sampling_rate).min()
            if lp_sampling_type == 'Max':
                df_plot = df_plot.resample(sampling_rate).max()
            else:
                df_plot = df_plot.resample(sampling_rate).first()
    except TypeError:
        error('Resampling Failed')

    if len(lp_y) > 2:
        fig = plotly_line_plot_many(df_plot, lp_y)
    else:

        fig = make_subplots(specs=[[{"secondary_y": True}]])  # Create figure with secondary y-axis
        x = df_plot.index
        y = df_plot[lp_y[0]]
        fig.add_trace(Line(x=x, y=y, name=lp_y[0] + " data", hoverinfo='skip'), secondary_y=False, )
        if len(lp_y) > 1:
            fig.add_trace(Line(x=x, y=df_plot[lp_y[1]], name=lp_y[1] + " data", hoverinfo='skip'),
                          secondary_y=True,)
        if len(lp_y) > 1:
            fig.update_layout(title_text="Line Plot for " + lp_y[0] + " & " + lp_y[1], width=400, height=600,
                              autosize=False)
        else:
            fig.update_layout(title_text="Line Plot for " + lp_y[0])
        fig.update_xaxes(title_text="DateTime", tickangle=-45, nticks=10)
        fig.update_yaxes(title_text="<b>primary</b> " + lp_y[0], secondary_y=False)
        if len(lp_y) > 1:
            fig.update_yaxes(title_text="<b>secondary</b> " + lp_y[1], secondary_y=True)
        # fig.update_traces(connectgaps=False)
    del df_plot
    return fig


def get_line_plot_form():
    with form("lp_form"):
        lp_options = list(session_state.df.columns)
        col_lp = columns([1, 5])
        with col_lp[0]:
            lp_sampling = selectbox(label='Select Sampling Rate',
                                       options=['1 second', '1 minute', '5 minutes', '10 minutes', '30 minutes',
                                                '1 hour', '6 hours'],
                                       help="Larger means Faster Plot")
            lp_sampling_type = selectbox(label='Select Sampling Type', options=['First', 'Mean', 'Min', 'Max'])
        with col_lp[1]:
            lp_y = multiselect(label='Select Y Axis  Features of Line Plot', options=lp_options,
                                  key='lp_plot',
                                  help='The Y-scaling of the Line plots is adjusted automatically',
                                  label_visibility="visible",
                                  max_selections=10)
        code("🕑 Estimated Plot Generation Time 10 seconds, for " + str(session_state.df.shape[0]) + " samples.",
                language=None)
        submitted_lp = form_submit_button("Generate Line Plot")
        if submitted_lp:
            if len(lp_y) < 1:
                error('Please Choose Features for Y Axis')
            else:
                with spinner('Generating Line Plot'):
                    code("Double click to reset the Zoom")
                    lp_plot = plotly_line_plot(lp_y, lp_sampling, lp_sampling_type)
                    plotly_chart(lp_plot, use_container_width=True)
### - - - - - - - - - - - - - - -

###---------- DENSITY PLOT TOOL

def df_density_plots_plotly_single(df=None, i='',cl=None, plot_style='Histogram'):

    if plot_style == "Histogram": fig = histogram(df, x=i, nbins=50)
    elif plot_style == "Violin": fig = violin(df, x=i, points=False)
    else: fig = box(df, x=i, points=False)

    return fig


def get_density_plot_form():
    with form("dp_form"):
        ld_options = session_state.df.select_dtypes(include=np_number).columns.tolist()
        selected_d = multiselect(label='Select Features to Get the Distribution Plots', options=ld_options,
                              key='ld_plot',
                              help='Density Plot Generation',
                              label_visibility="visible",
                              max_selections=4)
        pt_d = columns(2)
        with pt_d[0]:
            plot_type = radio(label='Plot Type', options=['Plain', 'clustered'], horizontal=True, index=0, disabled=True)
        with pt_d[1]:
            plot_style = radio(label='Plot Style', options=['Histogram', 'Violin', 'Box'], horizontal=True, index=0)

        write(
            'Density Plots show the Distribution of each Feature. ***x-axis***: Value Range and ***y-axis***: Density [max: 1]')
        code("Data below the 2nd and above the 98th Percentile are removed in order to preserve Plot Balance",
                language=None)
        if session_state.df.shape[0] > 200000:
            warning('Current Sample Size of ' + str(session_state.df.shape[0]) +
                       ' is too large. A representative subsample of 200,000 of the DataTable will be used instead.',
                       icon="⚠️")
        else:
            info('💡 Current Sample Size of ' + str(session_state.df.shape[0]) + ' is sufficient in order to use all the DataTable Samples for the Density Plot Generation.')
        submitted_dp = form_submit_button("Generate Density Plot")
        if submitted_dp:
            dp_cols = columns(4)
            if session_state.df.shape[0] > 150000:
                df_tmp = session_state.df.sample(150000)
            else:
                df_tmp = session_state.df
            if plot_type == 'Plain':
                cl = None
            # add more options with elf

            with spinner('Generating Density Plots...'):
                col_c = 0
                for i in selected_d: #session_state.df.select_dtypes(include=np_number).columns.tolist():
                    with dp_cols[col_c]:
                        plotly_chart(df_density_plots_plotly_single(df=df_tmp, i=i, cl=None, plot_style=plot_style), use_container_width=True)
                    col_c += 1
                    if col_c > 3:
                        col_c = 0
    return


@cache_data(show_spinner="Generating DataTable Statistics...")
def get_table_description():
    with container():
        markdown("<h6 style='text-align: left; color: #5a5b5e;margin-top:1em'>Numeric Features</h6>",
                    unsafe_allow_html=True)
        dataframe(session_state.df.describe().T, height=400, use_container_width=True)
        markdown("<h6 style='text-align: left; color: #5a5b5e;margin-top:1em'>Categorical Features</h6>",
                    unsafe_allow_html=True)
        if len(session_state.df.select_dtypes(include=['object']).columns) > 0:
            dataframe(session_state.df.describe(include='object').T, height=160, use_container_width=True)
        else:
            warning('No Categorical Features to show')
