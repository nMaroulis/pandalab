from streamlit import (markdown, file_uploader, columns, session_state, write, info, select_slider, button, warning, dataframe, spinner, date_input, time_input,
                       selectbox, container, radio, expander, text_input, toggle, rerun, number_input, dialog, error, empty as st_empty, form, form_submit_button, html, download_button, toast)
import warnings
from os import listdir, path, walk
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
from library.file_reader.file_loader import create_data_table, create_data_table_from_bookmark
from library.file_reader.dictionary_funcs import get_dictionaries
from datetime import datetime, time as dt_time


def load_uploaded_logs():

    markdown("<h6 style='text-align: center; color: #5a5b5e;'>Data Uploader from local Filesystem</h6>", unsafe_allow_html=True)

    uploaded_files = file_uploader(
        label="File Uploader",
        key="1",
        help="You can upload multiple files at once.",
        accept_multiple_files=True,
        label_visibility="hidden"
    )

    if len(uploaded_files) > 0:
        create_data_table(uploaded_files, 'upload', session_state['sampling_rate'], session_state['timestamp_label'],
                          session_state['timestamp_format'], session_state['sampling_type'],  session_state['csv_seperator'], session_state['csv_skiplines'], session_state['dictionary_choice'])
    else:
        session_state['file_uploading_options_container'] = st_empty()
        with session_state['file_uploading_options_container'].container():
            info(f""":page_facing_up: Supported File Formats are (***.csv***, ***.parquet***, ***.excel***, ***.mf4***, ***.dat***)""")
            with expander('**(1) Output Options:**', expanded=True):
                col20, col21 = columns(2)
                with col20:
                    session_state['sampling_type'] = radio('Resampling Type:', options=['first', 'mean', 'max'], help='first: Keep 1 sample of the whole second and remove the others. Mean: take the average from each sample of the second. Max: take the maximum sample within the second.', horizontal=True)
                    session_state['compress_columns'] = toggle('Compress Column Types [memory usage optimization]', value=False)
                with col21:
                    session_state['sampling_rate'] = select_slider(
                        'Select the Desired **Sampling Rate** of the Data',
                        options=['Original', '1000Hz', '100Hz', '10Hz', '1Hz', '0.1Hz', '0.01Hz'], value='1Hz')
                    write('Selected Sampling Rate', session_state['sampling_rate'])
            with expander('**(2) Input Options:**', expanded=True):
                col10, col11 = columns(2, gap="large")
                with col10:
                    session_state['timestamp_label'] = text_input('Label of the Datetime Column:',
                                                                  placeholder='Label of DateTime in the Datasets..')
                    session_state['data_imputation'] = toggle('Impute Data', value=True,
                                                              help="If certain numerical columns contain characters like >, <, +, -, remove them and just keep the Number")
                with col11:
                    session_state['timestamp_format'] = radio('Timestamp Format', options=['Unix', 'DateTime', 'None'],
                                                              index=0,
                                                              help='Unix Timestamp is in the format of 1633524480. DateTime: yyyy/mm/dd hh:mm:ss. If None then not DateTime will be created.',
                                                              horizontal=True)
            with expander('**(3) File Options:**', expanded=True):
                col00, col01 = columns(2)
                with col00:
                    session_state['datatable_name'] = text_input('DataTable Name:', placeholder='Write the desired DataTable name!')
                    session_state['csv_seperator'] = selectbox('CSV Seperator',
                                                               help="In case of CSV as uploaded File, please include the Seperator Symbol",
                                                               options=[',', ';', ' ', '|', 'tab'])
                with col01:
                    dict_options = get_dictionaries()
                    dict_options.insert(0, '<None>')
                    session_state['dictionary_choice'] = selectbox('Use Dictionary', options=dict_options)
                    session_state['csv_skiplines'] = number_input('CSV Skip Lines',
                                                               help="In case the CSV contains metadata in the beginning of the file, choose the number of lines to skip until the Headers start,",
                                                               min_value=0, max_value=10000, value=0)


def get_bookmark_archives(directory_path='archive/bookmarks'):
    dir_list = []
    if path.exists(directory_path):
        dir_list = []
        for directory in listdir(directory_path):
            d = path.join(directory_path, directory)
            if path.isdir(d):
                dir_list.append(d[d.rindex('/') + 1:])
        dir_list.sort()
    return dir_list


def load_bookmarks():
    markdown("<h6 style='text-align: center; color: #5a5b5e;'>Load Bookmark from local Archive</h6>", unsafe_allow_html=True)

    with container():

        directory_path = 'archive/bookmarks'
        dir_list = get_bookmark_archives(directory_path)
        dir_choice = selectbox(label='Choose Archive Directory:', options=dir_list)

        if len(dir_list) > 0:

            selected_dir = path.join(directory_path, dir_choice)
            if path.exists(selected_dir):
                bookmark_list = []
                for f in listdir(selected_dir):
                    bf = path.join(selected_dir, f)
                    if path.isfile(bf):
                        bookmark_list.append(bf[bf.rindex('/') + 1:])
                bookmark_list.sort()
            bookmark_choice = selectbox(label='Choose Bookmark to load:', options=bookmark_list)

            markdown("<hr style='text-align: left; width:15em; margin: 0em 0em; color: #5a5b5e'></hr>", unsafe_allow_html=True)

            if len(bookmark_list) < 1:
                button(label='Generate Data Table', disabled=True)
            else:
                generate_dt_from_bookmark = button(label='Generate Data Table', type='secondary')
                if generate_dt_from_bookmark:
                    session_state['df'] = create_data_table_from_bookmark(directory_path + '/' + dir_choice + '/' + bookmark_choice)
                    rerun()

def download_datatable_button():
    if button('Download Data Table 🔗', use_container_width=True):
        download_datatable_dialog()

@dialog('Datatable Download Manager', width='large')
def download_datatable_dialog():
    html("<br><h5 style='text-align: left; color: #5a5b5e;padding-bottom:0;'>Download DataTable</h5>")
    dt_download_status = False
    with form('Download DataTable Form', clear_on_submit=True):
        dt_name = text_input('File Name', value=session_state['datatable_name'],
                                placeholder='Insert file name here')
        file_type_choice = radio('File Type', options=['csv', 'parquet', 'mf4', 'excel'], horizontal=True,
                                    disabled=True)
        html("<br>")
        dt_download_submit = form_submit_button('Create Download Link 🔗')
        if dt_download_submit:
            if len(dt_name) < 1:
                dt_name = 'datatable_' + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            dt_download_status = True
    if dt_download_status:
        with spinner('Preparing Download Link...'):
            write('File name: ' + dt_name + '.csv')
            download_button(
                label="Download DataTable as CSV",
                data=session_state.df.to_csv().encode('utf-8'),
                file_name=dt_name + '.csv',
                mime='text/csv',
                type='primary'
            )
            toast('📄 File is ready to download.')


def save_bookmark_button():
    if button('Save to Bookmark', use_container_width=True):
        bookmark_datatable_dialog()


@dialog('Bookmark Manager', width='large')
def bookmark_datatable_dialog():
    html("<br><h5 style='text-align: left; color: #5a5b5e;padding-bottom:0;'>Save Bookmark to Server Disk</h5>")
    with form('Save DataTable to Bookmark Form'):
        col1, col2 = columns(2)
        with col1:
            dir_list = get_bookmark_archives('archive/bookmarks')
            arch_dir = selectbox(label='Choose Archive Directory:', options=dir_list)
        with col2:
            arch_dt_name = text_input('File Name', value=session_state['datatable_name'], placeholder='Insert file name here')

        info('💡 If file name is left empty, the filename will be automatically set to datatable_[current_datetime].')
        info('💡 The file type is automatically set to Parquet, since it\'s the optimal in terms of Efficiency and Performance.')
        html("<br>")
        arch_save_submit = form_submit_button('Save to Archive Catalog')
        if arch_save_submit:
            with spinner('Saving DataTable to Archive'):
                if len(arch_dt_name) < 1:
                    arch_dt_name = 'datatable_' + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                    write(arch_dt_name)
                try:
                    session_state.df.to_parquet('archive/bookmarks/' + arch_dir + '/' + arch_dt_name + '.parquet', compression='gzip')
                    toast('File Saved Successfully to Archive Catalog!', icon="✅ ")
                except Exception as e:
                    error('Something went wrong while saving File to Archive Catalog, Please report the following error to the administrator' + str(e))
                    print('LOG :: Bookmark save ::', e)
