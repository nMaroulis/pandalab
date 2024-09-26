from streamlit import (session_state, form, form_submit_button, caption, html, columns, number_input, warning,
                       divider, success, toggle, write, multiselect, radio, selectbox, pyplot, toast, select_slider, cache_data)
from numpy import median as np_median, mean as np_mean
from scipy.signal import savgol_filter
from matplotlib.pyplot import subplots as plt_subplots


def feature_transformation(tr_features, tr_feature_new, tr_feature_type):
    if len(tr_features) < 1:
        warning('Feature Transformation Failed, No Features chosen!')
    else:
        new_features = []
        for f in tr_features:
            if tr_feature_type == 'Diff [delta]':
                if tr_feature_new:
                    session_state.df[f+'_delta'] = session_state.df[f].diff()
                    new_features.append(f+'_delta')
                else:
                    session_state.df[f] = session_state.df[f].diff()
            elif tr_feature_type == 'Sum [Cumulative]':
                if tr_feature_new:
                    session_state.df[f+'_sum'] = session_state.df[f].sum()
                    new_features.append(f+'_sum')
                else:
                    session_state.df[f] = session_state.df[f].sum()
            else:  # Log
                if tr_feature_new:
                    session_state.df[f+'_log'] = session_state.df[f].log()
                    new_features.append(f+'_log')
                else:
                    session_state.df[f] = session_state.df[f].log()
        if tr_feature_new:
            success(str(len(new_features)) + ' ***new** Features were created successfully using the ' + tr_feature_type + ' Transformation')
            write(new_features)
        else:
            success('The following **' + str(len(tr_features)) + ' Features** were transformed successfully using the **' + tr_feature_type + ' Transformation**')
            write(tr_features)
        return


def get_savgol_signal(signal_series=None, savgol_window=300, mf_window=3000, savgol_polyorder=3, mf_mean_median='mean'):
    fig, ax = plt_subplots(figsize=(18, 6))

    savgol_window = savgol_window
    mf_window = mf_window

    savgol_ts = savgol_filter(signal_series.values.squeeze(), savgol_window, savgol_polyorder)

    moving_averages = []
    j = 0
    for i in range(1, len(savgol_ts)):
        if i % mf_window == 0:
            for k in range(mf_window):
                if mf_mean_median == 'mean':
                    moving_averages.append(np_median(savgol_ts[j * mf_window:i]))
                else:
                    moving_averages.append(np_mean(savgol_ts[j * mf_window:i]))
            j += 1

    # Fill missing values
    values_missing = len(savgol_ts) - len(moving_averages)
    if mf_mean_median == 'mean':
        final_window = np_mean(savgol_ts[len(moving_averages):len(savgol_ts)])
    else:
        final_window = np_median(savgol_ts[len(moving_averages):len(savgol_ts)])

    for k in range(values_missing):
        moving_averages.append(final_window)

    ax.plot(signal_series.values, label=signal_series.name)
    ax.plot(moving_averages, label='Smoothed ' + signal_series.name)
    ax.legend()
    return moving_averages, fig


# deprecated varsion in archived_funcs
def moving_filter(mf_features, savgol_window=300, mf_window=300, savgol_polyorder=3, mf_new_feature=True, mf_mean_median='mean', mf_type='Custom'):
    if len(mf_features) < 1:
        warning('Moving Filter Transformation Failed, No Features chosen!')
    else:
        new_features = []
        for f in mf_features:
            moving_averages, fig = get_savgol_signal(session_state.df[f], savgol_window, mf_window, savgol_polyorder, mf_mean_median)
            # # CREATE NEW OF CHANGE EXISTING
            if mf_new_feature:
                session_state.df[f+'_smoothed'] = moving_averages
                new_features.append(f+'_smoothed')
            else:
                session_state.df[f] = moving_averages

        write("Original & Smoothed " + f + " feature")
        pyplot(fig, use_container_width=True)
        # PRINT NEW FEATURES
        if mf_new_feature:
            write(str(len(new_features)) + ' **new** Feature(s) were created successfully using the Custom Savitzky-Golay Window Size + Moving Median Filter')
            write(new_features)
        else:
            success('The following ' + str(len(new_features)) + ' Features were transformed successfully using the ' + mf_type + ' Moving Filter')
            write(mf_features)
        return


def feature_transformation_form():
    with form("data_transform"):

        # TRANSFORMATION
        html("<h5 style='text-align: left; color: #787878;padding:0'>1. Transformation Methods</h5>")
        caption('Choose a transformation to be applied on selected columns.')
        html("<hr style='text-align: left; width:5em; margin: 0; color: #5a5b5e'></hr>")
        tr_features = multiselect('Features', options=list(session_state.df._get_numeric_data().columns), placeholder="Choose Features")
        tr_feature_new = toggle('Generate **new** Feature', value=False, help="If this option is not activated the original Feature will be overwritten, otherwise, a new Feature will be create e.g. Feature1 -> Feature1_delta")
        tr_feature_type = radio('Format Type', options=['Diff [delta]', 'Sum [Cumulative]', 'Log'], horizontal=True)
        divider()

        # MOVING FILTER
        html("<h5 style='text-align: left; color: #787878;padding:0'>2. Moving Filter</h5>")
        caption("The Filter starts by transforming the Feature using a **Savitzky–Golay filter**, which is a digital filter that can be applied to a set of data points for the purpose of smoothing the data, that is, to increase the precision of the data without distorting the feature tendency, through a convolution process. The output Signal is then fed into a **Moving Median Filter** to further **smooth** the Feature step-wise.")
        html("<hr style='text-align: left; width:5em; margin: 0; color: #5a5b5e'></hr>")
        radio('Moving Smoothing Method', options=['Custom', 'Savitzky-Golay filter', 'Digitization [Binning]', 'Exponential', 'Mean', 'Median', 'Min', 'Max', 'Wilder', 'Low Pass Filter'], horizontal=True, disabled=True)

        mf_features = multiselect("Select Feature(s)", options=list(session_state.df._get_numeric_data().columns), max_selections=1)
        mf_cols = columns(3)
        with mf_cols[0]:
            savgol_window = number_input('Savitzky-Golay Window Size', value=300, min_value=2, help="Savitzky-Golay Filter")
            mf_mean_median = radio('Calculation:', options=['mean', 'median'], horizontal=True)
            mf_feature_new = toggle('Generate **new** Feature', value=True)
        with mf_cols[1]:
            mf_window = number_input('Moving Average Window Size', value=300, min_value=2, help="Used after the Savitzky-Golay Filter")
        with mf_cols[2]:
            savgol_polyorder = select_slider('Savitzky-Golay Polynomial Order', help="The order of the polynomial used to fit the samples. polyorder must be less than window_length", options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], value=3)
            # select_slider('Exponential Smoothening Factor', help="Degree to which weight of observation decreases with Time", options=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], value=0.6, disabled=True)
        divider()

        # SCALING
        html("<h5 style='text-align: left; color: #787878;padding:0'>3. Feature Scaling</h5>")
        caption("Change the Scaling of selected Features")
        html("<hr style='text-align: left; width:5em; margin: 0; color: #5a5b5e'></hr>")
        fts_cols = columns([3, 1, 1])
        with fts_cols[0]:
            sc_features = multiselect("Features", options=list(session_state.df._get_numeric_data().columns))
        with fts_cols[1]:
            sc_min = number_input("Min. Value", value=0.0)
        with fts_cols[2]:
            sc_max = number_input("Max. Value", value=1.0)
        divider()

        # DATETIME
        html("<h5 style='text-align: left; color: #787878;padding:0'>4. DateTime Feature</h5>")
        caption('Choose which column contains the DateTime column and make it the Index of the DataTable.')
        html("<hr style='text-align: left; width:5em; margin: 0; color: #5a5b5e'></hr>")
        selectbox('DateTime Feature', options=list(session_state.df.columns))
        radio('Format Type', options=['Unix Timestamp', 'DateTime [s]', 'DateTime [ms]', 'DateTime [ns]'],
                 horizontal=True)
        divider()

        # LABEL
        html("<h5 style='text-align: left; color: #787878;padding:0'>5. Categorical Encoding</h5>")
        caption('Encode categorical (non-numerical) features, in order to be able to feed it in a model, or use it for an analysis.')
        caption('The default Label Encoding will replace String Values with Integers (e.g. "ABC" → 1)')
        html("<hr style='text-align: left; width:5em; margin: 0; color: #5a5b5e'></hr>")
        radio('Encoding Method', options=['Label', 'One-Hot', 'Ordinal'], horizontal=True, disabled=True)
        divider()

        write("Transformation Options:")
        # sidebar.toggle('1. Transformation')
        ftcols = columns(5)
        with ftcols[0]:
            transf_choice = toggle('(1) Transformation')
        with ftcols[1]:
            filter_choice = toggle('(2) Moving Filter')
        with ftcols[2]:
            scaling_choice = toggle('(3) Scaling')
        with ftcols[3]:
            toggle('(4) DateTime')
        with ftcols[4]:
            toggle('(5) Categorical Encoding')

        submitted_ft = form_submit_button("Transform DataTable")
        if submitted_ft:
            if transf_choice or filter_choice or scaling_choice:
                html("<h4 style='text-align: left; color: #787878;margin-top:1em'>Report:</h4>")
                if transf_choice:
                    html("<h6 style='text-align: left; color: #787878;margin-top:1em'>Feature Transformation</h6>")
                    feature_transformation(tr_features, tr_feature_new, tr_feature_type)
                if filter_choice:
                    html("<h6 style='text-align: left; color: #787878;margin-top:1em'>Moving Filtered Feature</h6>")
                    moving_filter(mf_features, savgol_window, mf_window, savgol_polyorder, mf_feature_new, mf_mean_median)
                if scaling_choice:
                    html("<h6 style='text-align: left; color: #787878;margin-top:1em'>Feature Scaling</h6>")
                    if len(sc_features) < 1:
                        warning('No Feature Chosen.')
                    else:
                        session_state.df[sc_features] = ((session_state.df[sc_features] - session_state.df[sc_features].min()) / (session_state.df[sc_features].max() - session_state.df[sc_features].min())) * (sc_max - sc_min) + sc_min
                        write('The following Features were scaled between ' + str(sc_min) + ' and ' + str(sc_max) + ' successfully!')
                        write(sc_features)
                cache_data.clear()  # CLEAR CACHE
                toast('✅ Data Transformation Task Finished Successfully!')
            else:
                warning('No option was chosen.')
