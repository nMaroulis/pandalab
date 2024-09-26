from streamlit import session_state, form, form_submit_button, columns, write, selectbox, radio, plotly_chart, pyplot,\
    image, warning, info, multiselect, markdown, code, spinner, error, fragment
from plotly.graph_objects import Figure, Histogram2dContour
from seaborn import pairplot
from plotly.express import parallel_coordinates, scatter_matrix, colors as px_colors


### PAIRPLOT TAB
def df_pair_plots(cols):
    # fig, ax = plt.subplots(1,1, figsize=(24, 24))
    sample_size = session_state.df.shape[0]
    if sample_size < 100000:
        pass
    else:
        sample_size = 100000

    pplot = pairplot(session_state.df.sample(sample_size)[cols], kind='hist')
    return pplot.fig


def df_pair_plots_plotly(cols):
    sample_size = session_state.df.shape[0]
    if sample_size < 100000:
        pass
    else:
        sample_size = 100000
    pplot = scatter_matrix(session_state.df.sample(sample_size)[cols], size_max=1, height=800)
    return pplot

@fragment
def get_pairplot_form():
    markdown("<h4 style='text-align: left; color: #48494B;'>Distribution Pairplot</h4>", unsafe_allow_html=True)
    with form("rpp_form"):
        write('The Relationship Pairplot display multiple pairwise bivariate distributions in a dataset. '
                 'The diagonal plots are the univariate plots, and this displays the relationship for the (n, 2) '
                 'combination of variables in a DataFrame as a matrix of plots.')
        dp_cols = columns(3)
        with dp_cols[1]:
            image('static/MultivariateNormal.png')
        rpp_features = multiselect(
            label='Choose up to 8 Features to generate the Pairplot for each Combinations between the Fetures',
            options=list(session_state.df.columns), max_selections=8, )
        if session_state.df.shape[0] > 100000:
            warning('Current Sample Size of ' + str(session_state.df.shape[
                                                           0]) + ' is too large. A representative subsample of 100,000 of the DataTable will be used instead.',
                       icon="⚠️")
        else:
            info('💡 Current Sample Size of ' + str(session_state.df.shape[
                                                          0]) + ' is sufficient in order to use all the DataTable Samples for the Pairplot Generation')
        markdown("<hr style='text-align: left; width:10em; margin: 1em 0em 1em; color: #5a5b5e'></hr>",
                    unsafe_allow_html=True)
        code(
            "🕑 Estimated Execution Time 40 seconds, for 8 features chosen",
            language=None)
        rpp_submit = form_submit_button("Generate Pairplot")
        if rpp_submit:
            if len(rpp_features) < 1:
                error('No Features chosen')
            else:
                with spinner('Generating Relationship Pairplot...'):
                    pyplot(df_pair_plots(rpp_features))
                    # plotly_chart(df_pair_plots_plotly(rpp_features), use_container_width=True)
### - - - - - - - - - - - - - - - - - - -


### CONTOUR MAP TAB
def plot_contour_map(rx, ry, rz, hfunc="avg"):

    fig = Figure(Histogram2dContour(
        x=session_state.df[rx],
        y=session_state.df[ry],
        z=session_state.df[rz],
        histfunc=hfunc,
        colorscale= 'Jet',  # 'RdYlBu_r', Edge, Jet
        colorbar={"title": rz},
        hoverinfo='skip',
        contours=dict(
            showlabels=True,
            labelfont=dict(
                family='Raleway',
                color='white'
            )
        ),
        # hoverlabel=dict(
        #     bgcolor='white',
        #     bordercolor='black',
        #     font=dict(
        #         family='Raleway',
        #         color='black'
        #     )
        # )
    ))
    # fig.update_traces(line_smoothing=1.3, selector = dict(type='histogram2dcontour'))
    fig.update_layout(title_text="Line Plot for x: " + rx + " & y: " + ry + " & z (color): " + rz, width=400, height=800,
                      autosize=False, xaxis_title=rx, yaxis_title=ry)
    return fig

@fragment
def get_contour_map_form():
    markdown("<h4 style='text-align: left; color: #48494B;'>Contour Map</h4>", unsafe_allow_html=True)
    write('The contour map consists of three features (x,y,z). The value of the z-axis will be denoted by the color, along with the value.')
    with form('cm_form'):
        cm_cols = columns(3)
        cm_options = list(session_state.df.columns)
        with cm_cols[0]:
            cm_x = selectbox(label='Select X Axis of Contour Map', options=cm_options, index=0, help="X-axis")
        with cm_cols[1]:
            cm_y = selectbox(label='Select Y Axis of Contour Map', options=cm_options, index=1, help="Y-axis")
        with cm_cols[2]:
            cm_z = selectbox(label='Select Z (Colour) Axis of Contour Map', options=cm_options, index=2, help="Z-axis")

        write("For the option ***count***, the contour map values are computed by counting the number of values "
              "lying inside each bin. If ***sum***, ***avg***, ***min***, ***max***, the contour map values are "
              "computed using the sum, the average, the minimum or the maximum of the values lying inside each bin respectively.")
        z_type = radio(label="Z Value Type", options=["count", "sum", "avg", "min", "max"], index=2, horizontal=True)

        cm_submitted = form_submit_button('Generate Contour Map')
        if cm_submitted:
            plotly_chart(plot_contour_map(cm_x, cm_y, cm_z, z_type), use_container_width=True)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def plot_parallel_coordinates(pcp_1, pcp_2, pcp_3, pcp_4, pcp_5, pcp_color):

    dimensions_list = []
    if pcp_1 != '<None>':
        dimensions_list.append(pcp_1)
        if pcp_2 != '<None>':
            dimensions_list.append(pcp_2)
            if pcp_3 != '<None>':
                dimensions_list.append(pcp_3)
                if pcp_4 != '<None>':
                    dimensions_list.append(pcp_4)
                    if pcp_5 != '<None>':
                        dimensions_list.append(pcp_5)

    smpl_rate = 1
    if session_state.df.shape[0] > 2000:
        smpl_rate = session_state.df.shape[0] // 2000
    fig = parallel_coordinates(session_state.df.iloc[::smpl_rate, :], color=pcp_color,
                               dimensions=dimensions_list,
                               color_continuous_scale=px_colors.diverging.Tealrose, height=820)
    return fig

@fragment
def get_pcp_form():
    markdown("<h4 style='text-align: left; color: #48494B;'>Parallel Coordinated Plot</h4>", unsafe_allow_html=True)
    write("Parallel coordinates is a visualization technique used to plot individual data elements across many performance measures. Each of the measures corresponds to a vertical axis and each data element is displayed as a series of connected points along the measure/axes.")
    with form('pcp_form'):
        pcp_cols = columns(5)
        pcp_options = list(session_state.df.columns)
        pcp_options.insert(0, '<None>')
        with pcp_cols[0]:
            pcp_1 = selectbox(label='Parallel axis 1', options=pcp_options)
            pcp_color = selectbox(label='Color Feature', options=pcp_options)
        with pcp_cols[1]:
            pcp_2 = selectbox(label='Parallel axis 2', options=pcp_options)
        with pcp_cols[2]:
            pcp_3 = selectbox(label='Parallel axis 3', options=pcp_options)
        with pcp_cols[3]:
            pcp_4 = selectbox(label='Parallel axis 4', options=pcp_options)
        with pcp_cols[4]:
            pcp_5 = selectbox(label='Parallel axis 5', options=pcp_options)

        info("ℹ️ The Data will be resampled to reduce Plot Noise")

        pmp_submitted = form_submit_button('Generate Parallel Coordinated Plot')
        if pmp_submitted:
            if pcp_1 == '<None>' or pcp_color == '<None>' or pcp_2 == '<None>':
                error("Fill out mandatory Fields Parallel Axis 1, Parallel Axis 2 and Color Feature")
            else:
                plotly_chart(plot_parallel_coordinates(pcp_1, pcp_2, pcp_3, pcp_4, pcp_5, pcp_color), use_container_width=True)
    return
