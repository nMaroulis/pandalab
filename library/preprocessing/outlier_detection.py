from streamlit import session_state, form, form_submit_button, caption, markdown, columns, number_input, image, divider, spinner, cache_data


def outlier_detection_form():
    with form("data_outlier"):
        markdown("<h5 style='text-align: left; color: #787878;padding:0'>Inter-Quantile Range</h5>",
                    unsafe_allow_html=True)
        caption("Values that have extreme Values, based on the Percentiles of the observed Data will be removed.")
        markdown("<hr style='text-align: left; width:5em; margin: 0; color: #5a5b5e'></hr>",
                    unsafe_allow_html=True)
        od_cols = columns(3)
        with od_cols[0]:
            image('static/iqr_dist.png', use_column_width=True)
        with od_cols[1]:
            number_input('Lower Percentile [%]', min_value=0, max_value=100, value=1)
        with od_cols[2]:
            number_input('Upper Percentile [%]', min_value=0, max_value=100, value=99)
        divider()
        submitted_od = form_submit_button("Filter DataTable")
        if submitted_od:
            with spinner('Removing Outliers from the DataTable'):
                session_state.df = session_state.df.drop([], axis=1)
            cache_data.clear()  # CLEAR CACHE
