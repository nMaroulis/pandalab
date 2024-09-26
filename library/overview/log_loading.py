import streamlit
from streamlit import html, file_uploader, columns, session_state, write, info, slider, select_slider, button, \
    number_input, selectbox, container, radio, expander, error, text_input
import warnings
from os import listdir, path  # walk
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
from library.overview.home_functions import create_data_table, create_data_table_from_bookmark, reset_df_and_page, datatable_dropna
from datetime import datetime
# from database.influxdb_connector import get_date_interval_from_influxdb, get_dataframe_from_influxdb, check_connection_to_influxdf


def load_uploaded_logs():
    html("<h5 style='text-align: center; color: #5a5b5e;'>Log File Uploader</h5>")
    uploaded_files = file_uploader(
        label="File Uploader",
        key="1",
        help="Be sure to be connected to the VPN in order to find NAS Folder",
        accept_multiple_files=True,
        label_visibility="hidden"
    )

    if len(uploaded_files) > 0:  # if uploaded_files is not None
        create_data_table(uploaded_files, 'upload', session_state['sampling_rate'], session_state['timestamp_label'])
    else:
        col1, col2, col3 = columns([1, 1, 2])
        with col1:
            session_state['datatable_name'] = text_input('DataTable Name:', placeholder='Write the desired DataTable name!')
        with col2:
            session_state['timestamp_label'] = text_input('Label of the Datetime Column:', placeholder='Label of DateTime in the Logs..')
        with col3:
            session_state['sampling_type'] = radio('Resampling Type:', options=['first', 'mean', 'max'], help='first: Keep 1 sample of the whole second and remove the others. Mean: take the average from each sample of the second. Max: take the maximum sample within the second.', horizontal=True)
            session_state['sampling_rate'] = select_slider(
                'Select the Desired **Sampling Rate** of the Data',
                options=['Unknown', '1000Hz', '100Hz', '10Hz', '1Hz'], value='1Hz')
            write('Selected Sampling Rate', session_state['sampling_rate'])
        # PARSING_COMPLETED = False
        info(f""":page_facing_up: Supported File Formats are (***.csv***, ***.mf4***, ***.dat***, ***.parquet***)""")
        info(f""":page_facing_up: The Filename must not contain the **.** character """)


def load_bookmarks():
    html("<h5 style='text-align: center; color: #5a5b5e;'>Load DataTables from Archive</h5>")
    with container():
        # FLEET CHOISE
        # archive_choice = radio(label="Source:", options=["PTDarchive", "Server Bookmarks", "AWS S3 Bucket"],
        #                           horizontal=True)

        bookmark_path = './archive/bookmarks'
        streamlit.subheader('Browse through Saved Bookmarked DataTables')
        html("<hr style='text-align: left; width:22em; margin: 0em 0em; color: #5a5b5e'></hr>")

        if path.exists(bookmark_path):
            bookmark_list = []
            for file in listdir(bookmark_path):
                d = path.join(bookmark_path, file)
                if path.isfile(d):
                    bookmark_list.append(d[d.rindex('/') + 1:])
            bookmark_list.sort()
        bookmark_choice = selectbox(label='Choose Bookmark to load:', options=bookmark_list)
        if len(bookmark_list) >= 1:
            if button(label='Generate Data Table', type='primary'):
                create_data_table_from_bookmark(bookmark_path + '/' + bookmark_choice)
        else:
            button(label='Generate Data Table', disabled=True)
