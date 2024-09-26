from streamlit import text, progress, session_state, spinner, rerun, error, success, markdown, cache_data
from library.file_reader.file_parser import create_df
from pandas import to_datetime, concat as pd_concat, read_parquet
import time


def create_data_table(files, upload_type='Server Disk',  sampling_rate='1Hz', datetime_label = ''):
    progr_text = text('Starting Parsing..')
    log_parsing_bar = progress(0)
    progress_counter = 0
    df_list = []
    session_state['sampling_rate'] = sampling_rate
    c = 0  # counter for valid number of logs
    tot = 0  # counter for total Number of Logs
    for uploaded_file in files:
        # CREATE DF
        df = create_df(uploaded_file, sampling_rate=sampling_rate, upload_type=upload_type, datetime_label=datetime_label)
        # APPEND TO LIST
        if df is not None:
            if len(list(df.columns)) > 10 and df.shape[0] > 10: # Basic Criteria to accept Log
                c += 1
                df_list.append(df.copy())
        else:
            pass

        tot += 1

        if upload_type == 'upload':
            uploaded_file.seek(0)
        progr_text.text('Parsing File ' + str(progress_counter + 1) + '/' + str(len(files)))
        log_parsing_bar.progress(progress_counter / len(files))
        progress_counter += 1
    progr_text.text('Finishing Data Preprocessing...')
    with spinner('Reading and Parsing Dataset Files...'):  # FINAL DF
        if len(df_list) > 1:
            # CONCAT LIST
            df = pd_concat(df_list, axis=0, ignore_index=True)
            print('home_page :: create_data_table :: Done Concat')
            del df_list  # free up space
            # SORT BY DATETIME
            df = df.sort_values(by='DateTime')
            # RESET INDEX
            # df.reset_index(inplace=True, drop=True)
            session_state['days'] = list(df['DateTime'].astype(str).str[0:10].unique())
            # OR
            df['DateTime'] = to_datetime(df['DateTime'])
            df.set_index('DateTime', inplace=True, drop=True)
            df.dropna(axis=1, how='all', inplace=True)

            session_state['logs_loaded'] = str(c) + '/' + str(tot)
            if df.shape[0] < 10:
                log_parsing_bar.empty()
                progr_text.empty()
                succ_msg = error('❌ Error Loading the Dataset - Corrupted Log Files. Refreshing page in 5 seconds.')
                time.sleep(4)
                succ_msg.empty()
                session_state.clear()
                rerun()
        else:  # in case of single File
            if len(df_list) <= 0:
                log_parsing_bar.empty()
                progr_text.empty()
                succ_msg = error('❌ Error Loading the Logs: Corrupted Log Files. Refreshing page in 5 seconds.')
                time.sleep(4)
                succ_msg.empty()
                session_state.clear()
                rerun()
            else:
                df = df_list[0].copy()
                del df_list
                df['DateTime'] = to_datetime(df['DateTime'])
                session_state['days'] = list(df['DateTime'].astype(str).str[0:10].unique())
                df.set_index('DateTime', inplace=True, drop=True)
                df.dropna(axis=1, how='all', inplace=True)
                session_state['logs_loaded'] = '1/1'

                # print(df.columns)
        print('home_page :: create_data_table :: Dataframe created successfully with', df.shape[0],'samples.')
        session_state['df'] = df  # SAVE DATAFRAME TO SESSION
        # st.session_state.null_status = 0 - for reseting cached plots

    log_parsing_bar.empty()
    progr_text.empty()
    succ_msg = success('Log Parsing Successful!')
    time.sleep(2)
    succ_msg.empty()
    cache_data.clear()  # clear everything previous
    # cache_resource.clear()
    rerun()


def create_data_table_from_bookmark(f):
    with spinner('Reading Bookmarked File'):
        # Read Bookmark File
        session_state.df = read_parquet(f)

        session_state['sampling_rate'] = '1Hz'
    succ_msg = success('Log Parsing Successful!')
    time.sleep(1)
    succ_msg.empty()
    cache_data.clear()  # clear everything previous
    # cache_resource.clear()
    rerun()


def reset_df_and_page():
    print('home_page :: reset_df_and_page')
    session_state.clear()
    rerun()
    return


def datatable_dropna():
    print('home_page :: datatable_dropna')
    session_state.df = session_state.df.dropna(axis=0)
    cache_data.clear()
    # st.session_state.null_status = 1 - reseting cached plots
    rerun()
    return



###### ---------------- FUNCTIONS
def _max_width_():
    max_width_str = f"max-width: 1800px;"
    markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )
# hash_funcs={matplotlib.figure.Figure: lambda _: None})
###### ---------------- FUNCTIONS END