from streamlit import markdown, session_state, caption, form, form_submit_button, success, selectbox, multiselect, \
    spinner, divider, columns, write, select_slider, toast, dataframe, progress, plotly_chart, cache_data, warning, button
from pandas import DataFrame
from scipy.spatial.distance import correlation as distance_corr
from pingouin import linear_regression, corr as pg_corr
from plotly.express import imshow
from numpy import unique as np_unique


def get_threshold(strictness='Moderate'):

    # the order is pearson, spearman, distance, biweight, percentage_bend
    strictness_dict = {
        'Very Low': [0.3, 0.3, 1.0, 0.3, 0.3],
        'Low':      [0.4, 0.4, 0.75, 0.4, 0.4],
        'Moderate': [0.5, 0.5, 0.6, 0.5, 0.5],
        'High':     [0.7, 0.7, 0.35, 0.7, 0.7],
        'Extreme':  [0.9, 0.9, 0.2, 0.9, 0.9]
    }

    return strictness_dict.get(strictness)


def get_reg_correlation_threshold(df_size, strictness='Moderate'):
    size_th = df_size // 10
    if size_th < 0:
        size_th = 1
    strictness_dict = {
        'Very Low': size_th*3,
        'Low':      size_th*2,
        'Moderate': size_th,
        'High':     size_th,
        'Extreme': 1
    }
    reg_th = strictness_dict.get(strictness)
    return reg_th


def highlight_row(row, selected_features_list):
    css_class = ''
    if row['Feature'] in selected_features_list:
        css_class = 'background-color: #d1ffd1'
    return [css_class] * len(row)


def get_bad_quality_columns(df):
    bad_cols = list(df.loc[:, df.isnull().mean() >= 0.5].columns)
    return bad_cols


def get_pearson_corrs(df, features, target=''):
    corrs = []
    for i in features:
        corrs.append(abs(round(df[i].corr(df[target]), 3)))
    return corrs


def get_spearman_corrs(df, features, target=''):
    corrs = []
    for i in features:
        corrs.append(abs(round(df[i].corr(df[target], method='spearman'), 3)))
    return corrs


def get_distance_corrs(df, features, target=''):
    corrs = []
    for i in features:
        d_df = df[[i, target]].dropna()
        dcor = distance_corr(d_df[i].values, d_df[target].values, w=None, centered=True)
        if dcor >= 1.8:  # because high distance means anti-correlation
            dcor = 2 - dcor
        corrs.append(abs(dcor))
    # del d_df
    return corrs


def get_regression_coefs(df, features, target='', corr_df=None):
    reg_df = linear_regression(df[features], df[target], coef_only=False, add_intercept=False, remove_na=True)  # Multiple linear regression
    # 1. Remove features with p-value > 0.05
    # 2. Study Coefficients
    # 3. Sort based on Coefficients, and t-value
    reg_df['Regression Coefficient (abs)'] = abs(reg_df['coef'])  # get absolute coef values
    reg_df.rename(columns={"names": "Feature"}, inplace=True)
    corr_df = DataFrame.merge(corr_df, reg_df[['Feature', 'Regression Coefficient (abs)']], on='Feature', how='left')
    return corr_df


def get_robust_correlations(df, features, target='', type='bicor'):
    corrs = []
    for i in features:
        try:
            bcor = pg_corr(df[i].values, df[target].values, method=type)
            corrs.append(abs(float(bcor['r'])))
        except AssertionError:
            corrs.append(None)
    return corrs


def selection_form():
    write(
        'The **Correlation Pipeline** calculates the following correlations between **every Feature** in the DataTable and the **Target Feature**. Then all the correlation scores are combined and sorted in order to choose the most correlated, using diverse methods.')
    write('Correlation Pipeline per **Depth** of Analysis')
    fs_cols1 = columns(4, gap='small')
    with fs_cols1[0]:
        write("**Low:**")
        write("- Pearson Correlation")
        write("- Spearman Correlation")
        write("- Distance Correlation")
        write("- Biweight Midcorrelation")
    with fs_cols1[1]:
        write("**Medium:**")
        write("+ Percentage Bend Correlation")
        write("- Regression Coefficients")
    with fs_cols1[2]:
        write("**High:**")
        write("- Skipped Correlation")
        write("- Kendall Rank Correlation")
        write("- Shepherd Pi Correlation")
    with fs_cols1[3]:
        write("**Extreme:**")
        write("- MIC")
        write("- Random Forest Feature Importance")

    with form("feature_selection"):
        markdown("<h5 style='text-align: left; color: #787878;padding:0'>Choose Model Target</h5>",
                    unsafe_allow_html=True)
        caption(
            'The Algorithm will find and keep the Features that have the highest correlation with the indicated Target Feature')
        markdown("<hr style='text-align: left; width:5em; margin: 0; color: #5a5b5e'></hr>",
                    unsafe_allow_html=True)

        fs_cols = columns([1, 2])
        with fs_cols[0]:
            fs_target = selectbox('**Target Feature**', options=list(session_state.df._get_numeric_data().columns))
        with fs_cols[1]:
            protected_cols = multiselect('Features to Protect (even if low Correlation is found)',
                                         options=list(session_state.df.columns))

        divider()
        write("Higher Levels of Depth for the Correlation Analysis, will include more correlation algorithms and formulas.")
        depth_of_analysis = select_slider('Depth of Analysis', ['Low [~45 sec]', 'Medium [~2 min]', 'High [~10 min]', 'Extreme [~30 min]'], value='Medium [~2 min]')

        divider()
        write("How Strict are the correlation thresholds, that the algorithm will take into account in order to choose which features to remove")
        removal_strictness = select_slider('Removal Strictness', ['Very Low', 'Low', 'Moderate', 'High', 'Extreme'], value='Moderate')

        submitted_fs = form_submit_button("Start Algorithm")
        if submitted_fs:

            if depth_of_analysis == 'Low [~45 sec]':
                prg_end = 4
            elif depth_of_analysis == 'Medium [~2 min]':
                prg_end = 6
            elif depth_of_analysis == 'High [~10 min]':
                prg_end = 6
            elif depth_of_analysis == 'Extreme [~30 min]':
                prg_end = 6
            else:
                prg_end = 6

            write('Correlation Analysis Progress..')
            progress_bar = progress(0)

            # GET COLUMNS THAT CONTAIN A LOT OF NULL AND REMOVE THEM
            bad_cols = get_bad_quality_columns(session_state.df)
            all_numeric_features = list(session_state.df._get_numeric_data().drop(fs_target, axis=1).columns)
            for bc in bad_cols:
                if bc in all_numeric_features:
                    all_numeric_features.remove(bc)
            corr_df = DataFrame(all_numeric_features, columns=["Feature"])
            final_feature_set = []
            # PEARSON
            with spinner(f'1/{prg_end} Calculating Pearson Correlations'):
                corr_df["Pearson Correlation"] = get_pearson_corrs(session_state.df, all_numeric_features, fs_target)
                final_feature_set = corr_df[corr_df["Pearson Correlation"] >= get_threshold(removal_strictness)[0]]["Feature"].to_list()
            progress_bar.progress(1/prg_end)
            #SPEARMAN
            with spinner(f'2/{prg_end} Calculating Spearman Correlations'):
                corr_df["Spearman Correlation"] = get_spearman_corrs(session_state.df, all_numeric_features, fs_target)
                final_feature_set.extend(corr_df[corr_df["Spearman Correlation"] >= get_threshold(removal_strictness)[1]]["Feature"].to_list())
            progress_bar.progress(2/prg_end)
            # DISTANCE CORRELATION
            with spinner(f'3/{prg_end} Calculating Distance Correlations'):
                corr_df["Distance Correlation"] = get_distance_corrs(session_state.df, all_numeric_features, fs_target)
                final_feature_set.extend(corr_df[corr_df["Distance Correlation"] <= get_threshold(removal_strictness)[2]]["Feature"].to_list())
            progress_bar.progress(3/prg_end)
            # BIWEIGHT CORRELATION
            with spinner(f'4/{prg_end} Calculating Biweight Midcorrelations'):
                corr_df["Biweight Midcorrelation"] = get_robust_correlations(session_state.df, all_numeric_features, fs_target, 'bicor')
                final_feature_set.extend(corr_df[corr_df["Biweight Midcorrelation"] >= get_threshold(removal_strictness)[3]]["Feature"].to_list())
            progress_bar.progress(4 / prg_end)


            if depth_of_analysis != 'Low [~45 sec]':
                # REGRESSION COEFFICIENTS
                with spinner(f'5/{prg_end} Calculating Regression Coefficients'):
                    corr_df = get_regression_coefs(session_state.df, all_numeric_features, fs_target, corr_df)
                    # final_feature_set.extend(corr_df[corr_df["Regression Coefficient (abs)"] >= 0.4]["Feature"].to_list())
                    reg_th = get_reg_correlation_threshold(
                        corr_df[corr_df["Regression Coefficient (abs)"] > 0].shape[0], removal_strictness)
                    final_feature_set.extend(corr_df[corr_df["Regression Coefficient (abs)"] > 0].nlargest(reg_th,
                                                                                                           'Regression Coefficient (abs)')[
                                                 'Feature'].to_list())
                progress_bar.progress(5 / prg_end)
                # PERCENTAGE BEND CORRELATION
                with spinner(f'6/{prg_end} Calculating Percentage Bend Correlation'):
                    corr_df["Percentage Bend Correlation"] = get_robust_correlations(session_state.df, all_numeric_features, fs_target, 'percbend')
                    final_feature_set.extend(corr_df[corr_df["Percentage Bend Correlation"] >= get_threshold(removal_strictness)[4]]["Feature"].to_list())
                progress_bar.progress(6/prg_end)
            # with spinner('Calculating Shepherd Pi Correlation'):
            #     corr_df["Shepherd Pi Correlation"] = get_robust_correlations(session_state.df, all_numeric_features, fs_target, 'shepherd')
            # with spinner('Calculating Skipped Correlation'):
            #     corr_df["Skipped Correlation"] = get_robust_correlations(session_state.df, all_numeric_features, fs_target, 'skipped')

            # FINAL LIST OF SELECTED FEATURES
            final_feature_set.append(fs_target)  # add target
            final_feature_set = list(np_unique(final_feature_set))  # keep unique
            for c in protected_cols:
                if c not in final_feature_set:
                    final_feature_set.append(c)

            # CREATE NEW DATAFRAME
            session_state['low corr features'] += session_state.df.shape[1] - len(final_feature_set)

            session_state.df = session_state.df[final_feature_set]
            success('Correlation algorithm finished successfully. Least Correlated Features were removed successfully.')
            toast('✅ Data Selection Task Finished Successfully!')

            markdown("<h5 style='text-align: left; color: #787878;padding:0'>Feature Selection Results</h5>",
                     unsafe_allow_html=True)
            write("> **" + str(len(final_feature_set)) + "** Features were found above the ***Correlation Threshold***")

            fs_cols2 = columns([2, 1], gap='small')
            with fs_cols2[0]:
                dataframe(corr_df.style.apply(lambda row: highlight_row(row, final_feature_set), axis=1))
                if corr_df[corr_df['Pearson Correlation'] >= 0.98].shape[0] > 0:
                    warning(
                        'The following Features showed a really high Linear Correlation with ***' + fs_target +
                        '***, which may indicate a **duplicate** Feature. If you wish to remove them manually, visit the Data Cleaning Tab for a Manual Removal',
                        icon="⚠️")
                    write(corr_df[corr_df['Pearson Correlation'] >= 0.98]['Feature'].to_list())
                if len(bad_cols) > 0:
                    warning('The following Features were NOT considered due to many empty values', icon="⚠️")
                    write(bad_cols)

            with fs_cols2[1]:
                correlations = corr_df.corr(method='kendall', numeric_only=True)
                fig = imshow(correlations, text_auto='.2f', aspect="auto")
                plotly_chart(fig, use_container_width=True)

            cache_data.clear()  # CLEAR CACHE
    return
