from streamlit import session_state, form, form_submit_button, write, caption, markdown, number_input, divider, code, \
    multiselect, toast, columns, spinner, success, cache_data, expander, table, warning, toggle, radio
from numpy import triu, ones  # , bool as np_bool


def remove_multicollinearity(p_th=0.95):

    correlations = session_state.df.corr(method='pearson', numeric_only=True).abs()
    upper_tr = correlations.where(triu(ones(correlations.shape), k=1).astype('bool'))
    to_drop = [column for column in upper_tr.columns if any(upper_tr[column] > p_th)]
    return to_drop


def remove_multicollinearity_with_protection(p_th=0.98, protected_columns=None):

    correlations = session_state.df.corr(method='pearson', numeric_only=True).abs()
    upper_tr = correlations.where(triu(ones(correlations.shape), k=1).astype('bool'))
    print(upper_tr.shape)
    res = [column for column in upper_tr.columns if any(upper_tr[column] > p_th) and column not in protected_columns]

    for column in protected_columns:
        for i, v in upper_tr[column].items():
            print(i, v)
            if v > p_th:
                if i not in res and i not in protected_columns:
                    res.append(i)
    # to_drop = []
    # for column in upper_tr.columns:
    #     correlated_columns = upper_tr.index[upper_tr[column] > p_th].tolist()
    #     for correlated_column in correlated_columns:
    #         if correlated_column not in protected_columns:
    #             to_drop.append(correlated_column)
    # # remove duplicates
    # res = []
    # [res.append(x) for x in to_drop if x not in res]
    return res


def remove_static_features(df, std_th=0.01):

    # cols_to_drop = df.std()[df.std() <= std_th].index.values
    cols_to_drop = df.columns[df.apply(lambda x: x.nunique(dropna=True) == 1)].tolist()
    # all_null_cols = df.columns[df.isnull().all()].tolist()
    # cols_to_drop.extend(all_null_cols)

    return cols_to_drop


def filtering_form():
    with form("data_filter"):
        markdown("<h5 style='text-align: left; color: #787878;padding:0'>1. Handle Static</h5>",
                    unsafe_allow_html=True)

        caption(
            'Static Features are the Features that have no variation, hence, they have the same Value during each DataTable sample.')
        markdown("<hr style='text-align: left; width:5em; margin: 0; color: #5a5b5e'></hr>", unsafe_allow_html=True)

        code("Each Value is first normalized between [0,1] and the Standard Deviation is calculated. Also the unique value vector of each Feature is calculated.", language=None)
        radio('Method', options=['Standard Deviation', 'One Unique Value'], disabled=True, horizontal=True)
        number_input('Standard Deviation Threshold', min_value=0.0, max_value=1.0, value=0.01, disabled=True)
        divider()
        markdown("<h5 style='text-align: left; color: #787878;padding:0'>2. Handle Multicollinearity </h5>",
                    unsafe_allow_html=True)
        caption(
            'Remove Features that have a really High correlation between them. These features provide the same Information, therefore, only one is sufficient.')
        markdown("<hr style='text-align: left; width:5em; margin: 0; color: #5a5b5e'></hr>", unsafe_allow_html=True)
        df_col1, df_col2 = columns([1, 3])
        with df_col1:
            autocorr_threshold = number_input('Pearson Correlation Threshold', min_value=0.0, max_value=1.0, value=0.98)
        with df_col2:
            protected_columns = multiselect('Features to Protect', options=list(session_state.df.columns))
        divider()
        markdown("<h5 style='text-align: left; color: #787878;padding:0'>3. Manually Remove Columns</h5>",
                    unsafe_allow_html=True)
        caption('Remove entire columns based on Feature names')
        markdown("<hr style='text-align: left; width:5em; margin: 0; color: #5a5b5e'></hr>", unsafe_allow_html=True)
        manual_remove_cols = multiselect('Features to Remove', options=list(session_state.df.columns))
        divider()
        write("Filter Options:")
        fcols = columns(5)
        with fcols[0]:
            static_choice = toggle('(1) Static')
        with fcols[1]:
            autocorr_choice = toggle('(2) Multi-Colinearity')
        with fcols[2]:
            manual_choice = toggle('(3) Manual')
        with fcols[3]:
            categorical_choice = toggle('(*) Remove Categorical')

        submitted_ff = form_submit_button("Filter DataTable")
        if submitted_ff:
            with spinner('Performing Data Filtering on the DataTable'):
                if manual_choice:
                    with spinner('Removing User-defined Features'):
                        session_state.df = session_state.df.drop(manual_remove_cols, axis=1)
                if static_choice:
                    with spinner('Removing Static Features'):
                        cols_to_drop = remove_static_features(session_state.df[session_state.df.columns]) # ._get_numeric_data()
                        session_state.df = session_state.df.drop(cols_to_drop, axis=1)
                if autocorr_choice:
                    with spinner('Removing Multi-Collinearity from the Features'):
                        if len(protected_columns) < 1:
                            cols_to_drop1 = remove_multicollinearity(autocorr_threshold)
                            # cols_to_drop1 = remove_multicollinearity_with_protection(autocorr_threshold, [])
                        else:
                            print('Protection')
                            cols_to_drop1 = remove_multicollinearity_with_protection(autocorr_threshold, protected_columns)
                        session_state.df = session_state.df.drop(cols_to_drop1, axis=1)
                if categorical_choice:
                    cat_cols = session_state.df.select_dtypes(include=['object']).columns
                    session_state.df = session_state.df.drop(cat_cols, axis=1)

                if manual_choice or static_choice or autocorr_choice or categorical_choice:
                    success('DataTable Filtered Successfully, based on chosen methods.')
                    with expander('Report', expanded=True):
                        fr_cols = columns(2)
                        with fr_cols[0]:
                            if static_choice:
                                write(len(cols_to_drop), 'Features were removed because they were **Static**')
                                table(cols_to_drop)
                                session_state['filtered features'][0] += len(cols_to_drop)
                            if manual_choice:
                                write(len(manual_remove_cols), 'Features were Manually removed')
                                session_state['filtered features'][2] += len(manual_remove_cols)
                                table(manual_remove_cols)
                        with fr_cols[1]:
                            if autocorr_choice:
                                cols_to_drop1.sort()
                                write(len(cols_to_drop1), 'Features were removed because of high **Linear Correlation** with other Independent Features')
                                table(cols_to_drop1)
                                session_state['filtered features'][1] += len(cols_to_drop1)
                            if categorical_choice:
                                write(len(cat_cols), 'Categorical Features were removed')
                                session_state['filtered features'][3] += len(cat_cols)
                                table(cat_cols)
                    cache_data.clear()  # CLEAR CACHE
                    toast('✅ Data Filtering Task Finished Successfully!')
                else:
                    warning('No option was chosen.')
