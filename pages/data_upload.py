import warnings
import streamlit as st
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
from library.file_reader.file_load_form import load_uploaded_datasets, load_bookmarks, download_datatable_button, save_bookmark_button
from library.overview.overview_plots import get_feature_status_pie_chart, get_feature_null_pie_chart, get_feature_descriptive_analytics
from library.overview.overview_tools import get_line_plot_form, get_density_plot_form, get_table_description
from library.overview.navigation import page_header
from PIL import Image
from library.overview.overview_plots import get_missing_values

from database.db_client import init_training_db, fetch_all

page_header()
# init_training_db()
# st.write(fetch_all())

_, img_col, _ = st.columns([2, 1, 2])
image = Image.open('static/logo.png')
with img_col:
    st.markdown("<br>", unsafe_allow_html=True)
    st.image(image, use_column_width=True)


if 'df' in st.session_state:
    st.markdown("<h3 style='text-align: left; color: #373737; padding: 0; margin: 0.5em 0 1em 0'>DataTable Overview</h3>", unsafe_allow_html=True)
else:
    # st.markdown("<h2 style='text-align: center; color: #373737;'>File Loader</h2>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; color: #48494B;'>Each module below provides a different interface to load Data to the Dashboard</h6>", unsafe_allow_html=True)

if 'df' not in st.session_state:  # if no File is Loaded
    upload_col, bookmark_col = st.tabs(["File Uploader", "Bookmark Archive"])
    with upload_col:
        load_uploaded_datasets()
    with bookmark_col:
        load_bookmarks()
else:  # if Dataframe is populated
    # st.markdown("<h4 style='text-align: center; color: #48494B;'>Data Table Overview</h4>", unsafe_allow_html=True)

    if'navigation_status' in st.session_state:
        if st.session_state['navigation_status'] == 'first_load':
            st.toast('Data loaded successfully', icon="✅")
            st.session_state['navigation_status'] = 'datatable'

    cols = st.columns(3)
    with cols[0]:
        st.markdown(f"""<br><br>""", unsafe_allow_html=True)
        st.metric('Number of columns', st.session_state.df.shape[1], (st.session_state.df.shape[1] - st.session_state['initial features']))
        st.metric('Number of Samples', st.session_state.df.shape[0], (st.session_state.df.shape[0] - st.session_state['initial samples']))

        memory_in_bytes = st.session_state.df.memory_usage(index=True).sum()
        if memory_in_bytes < 1024 ** 2:
            st.metric('DataTable size in Memory', f'{memory_in_bytes / 1024:.1f} KBs')
        elif memory_in_bytes < 1024 ** 3:
            st.metric('DataTable size in Memory', f'{memory_in_bytes / (1024 ** 2):.2f} MBs')
        else:
            st.metric('DataTable size in Memory', f'{memory_in_bytes / (1024 ** 3):.3f} GBs')

    with cols[1]:
        get_feature_null_pie_chart()
    with cols[2]:
        get_feature_status_pie_chart()

    num_cols = len(st.session_state.df._get_numeric_data().columns)

    st.write('Numerical Features', num_cols, 'Categorical Features', (st.session_state.df.shape[1] - num_cols))

    print('hope_page :: View :: Dataframe Loaded')
    st.markdown("""
    <style>
    .big-title-font {
        color: #373737;
        text-align: center !important;
    }
    .log-font {
        font-size:10px !important; color: #373737;
    }
    </style>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Data Table Statistics", "Distribution Plot", "Line Plots"])
    ### Two ways to reload a plot that has already been created - 1. save to RAM session cache and reload or save and load file
    with tab1:
        st.markdown("<h5 style='text-align: left; color: #5a5b5e;margin-top:2em'>Data Table Statistics</h5>",
                    unsafe_allow_html=True)
        st.caption('The Statistics for Numeric Features include the number of non-empty values, the **average**, '
                   '**standard deviation**, **min**, **25th percentile**, **median** (50%), **75th percentile** and **max**.')
        get_table_description()

        with st.container():
            st.markdown("<h5 style='text-align: left; color: #48494B;margin-top:3em'>Data Table</h5>", unsafe_allow_html=True)
            st.caption('Below a Sample of the DataTable is shown. Choose the Range of the DataTable Samples to be shown. Large DataTable Sample sizes might cause performance issues.')
            if st.session_state.df.shape[0] > 500:
                range_from, range_to = st.slider("Select Sample Range:", 0, st.session_state.df.shape[0], (0, 500))
                if (range_to - range_from) > 100000:
                    st.warning('Cannot display more than 100,000 rows.')
                else:
                    st.dataframe(st.session_state.df[range_from:range_to], height=400, use_container_width=True)
            else:
                _ = st.slider("Select Sample Range:", 0, st.session_state.df.shape[0], (0, st.session_state.df.shape[0]), disabled=True)
                st.dataframe(st.session_state.df, height=400, use_container_width=True)

        st.markdown("<h5 style='color: #525355;'>Column Health</h5>", unsafe_allow_html=True)
        st.code("The following barplot shows the percentage of valid, invalid and missing values for each Feature in the Data Table", language=None)
        if st.session_state.df.shape[1] > 40:
            st.warning("Too many columns in the datatable, cannot show plot")
        else:
            st.bar_chart(get_missing_values())


        with st.container():
            st.markdown("<h5 style='color: #525355; margin-top:3em'>Feature Analyzer 🔎</h5>", unsafe_allow_html=True)
            all_cols = list(st.session_state.df.columns)
            all_cols.insert(0, '<None>')
            column_selected_info = st.selectbox('Choose Feature to get more Information:', options=all_cols)
            get_feature_descriptive_analytics(column_selected_info)

        with st.expander(label='Data Table Report', expanded=False):
            st.button(label='Generate Detailed DataTable Report', disabled=True)
            # pr = ProfileReport(st.session_state.df)
            # st_profile_report(pr)
    with tab2:
        st.subheader("Density - Distribution Plot")
        get_density_plot_form()
    with tab3:
        st.subheader("Line Plot")
        get_line_plot_form()

    with st.sidebar:
        download_datatable_button()
    with st.sidebar:
        save_bookmark_button()