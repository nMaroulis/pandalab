from streamlit import button, form, form_submit_button, write, markdown, caption, divider, session_state, radio, \
    slider, spinner, success, dataframe, table, columns, warning, toggle, toast, cache_data, multiselect
from numpy import inf as np_inf, nan as np_nan, isinf as np_isinf
from pandas import DataFrame as pd_dataframe


def clean_faulty_features(features_to_convert, features_to_clean, clean_all, f_clean_options):

    for f in features_to_convert:
        session_state.df[f] = session_state.df[f].apply(lambda x: x if isinstance(x, float) else np_nan)

    if clean_all:
        features_to_clean = session_state.df._get_numeric_data().columns.to_list()
    for f in features_to_clean:
        session_state.df[f].replace([np_inf, -np_inf], np_nan, inplace=True)

    return


def cleaning_form():
    if button('Run Diagnostic', type='primary'):
        with spinner('Finding Null Values in DataTable'):

            col_name, num_nulls, num_infs, dtypes = [], [], [], []
            for col in session_state.df.columns:
                col_name.append(col)
                num_nulls.append(session_state.df[col].isnull().sum())
                if session_state.df[col].dtype == 'object':
                    num_infs.append(None)
                else:
                    num_infs.append(np_isinf(session_state.df[col]).sum())
                dtypes.append(session_state.df[col].dtype)
            d = {'Feature': col_name, 'Type': dtypes, 'Empty Values': num_nulls, 'Faulty Numerical': num_infs}
            d_df = pd_dataframe(data=d).set_index('Feature', drop=True)
            dataframe(d_df, use_container_width=True)

    with form("data_cleaning"):
        markdown("<h5 style='text-align: left; color: #787878;padding:0'>1. Handle Empty Values</h5>",
                    unsafe_allow_html=True)
        markdown("<hr style='text-align: left; width:5em; margin-top: 1em; color: #5a5b5e'></hr>", unsafe_allow_html=True)

        markdown("<h6 style='text-align: left; color: #787878;padding:0'>1.1 Replace All</h6>",
                    unsafe_allow_html=True)
        caption('Choose how to treat Empty/Null values in the DataTable.')
        radio('Method', options=['Drop', 'Forward Fill', 'Backward Fill'], horizontal=True, disabled=True)
        warning('⚠️ If certain columns contain many Null values, the Drop option could delete most of the DataTable.')
        markdown("<br>", unsafe_allow_html=True)
        markdown("<h6 style='text-align: left; color: #787878;padding:0'>1.2 Replace Features that contain more than n% Empty Values</h6>",
                    unsafe_allow_html=True)
        null_percentage = slider('Choose Percentage', min_value=1, max_value=100, value=80)
        divider()

        markdown("<h5 style='text-align: left; color: #787878;padding:0'>2. Restore Unhealthy Features</h5>",
                    unsafe_allow_html=True)
        markdown("<hr style='text-align: left; width:5em; margin-top: 1em; color: #5a5b5e'></hr>", unsafe_allow_html=True)

        markdown("<h6 style='text-align: left; color: #787878;padding:0'>2.1 Convert Categorical to Numerical</h6>",
                    unsafe_allow_html=True)
        caption('Some **Numerical** Features are maybe **translated as Categorical**, due to some bad values. Convert them using the following form.')
        features_to_convert = multiselect('Categorical Features', options=session_state.df.select_dtypes(include=['object']).columns.to_list())
        markdown("<br>", unsafe_allow_html=True)

        markdown("<h6 style='text-align: left; color: #787878;padding:0;'>2.2 Clean Numerical Features</h6>",
                    unsafe_allow_html=True)
        caption('Some **Numerical** Features might contain **Faulty Values**, that make them unusable. Clean them using the following form.')
        features_to_clean = multiselect('Categorical Features', options=session_state.df._get_numeric_data().columns.to_list())
        clean_all = toggle('Clean all', value=False)
        f_clean_options = radio('Replace Method', options=['Empty', 'Drop', 'Forward Fill', 'Backward Fill'], disabled=True, horizontal=True)
        divider()

        write('Cleaning Options:')
        ccols = columns(4)
        with ccols[0]:
            rem_all = toggle('(1.1) Remove all Null')
        with ccols[1]:
            rem_perc = toggle('(1.2) Percentage of Null in Feature')
        with ccols[2]:
            cln_feature = toggle('(2) Restore Unhealthy Features')

        submitted_fc = form_submit_button("Clean DataTable")
        if submitted_fc:
            with spinner('Cleaning DataTable...'):

                if rem_perc:
                    write("Features that contain > ", null_percentage, "% of Null Values:")
                    null_cols_to_remove = list(session_state.df.loc[:, session_state.df.isnull().mean() > (null_percentage/100)].columns)
                    table(null_cols_to_remove)
                    session_state.df = session_state.df.drop(null_cols_to_remove, axis=1)
                    session_state['cleaned features'] += len(null_cols_to_remove)
                if rem_all:
                    session_state.df = session_state.df.dropna()
                if cln_feature:
                    clean_faulty_features(features_to_convert, features_to_clean,clean_all, f_clean_options)
                if rem_all or rem_perc or cln_feature:
                    cache_data.clear()  # CLEAR CACHE
                    success('DataTable Cleaned Successfully, based on chosen methods.')
                else:
                    warning('No option was chosen.')
                    return
            toast('✅ Cleaning Task Finished Successfully!')
