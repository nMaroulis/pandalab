from streamlit import text, progress, session_state, spinner, rerun, error, success, toast, cache_data, empty
from library.file_reader.file_parser import create_df
from pandas import to_datetime, concat as pd_concat, read_parquet, api
import time
import gc
from library.file_reader.file_loader_funcs import initialize_session_state_vars, clear_loading_forms


def create_data_table(files, upload_type='upload',  sampling_rate='1Hz', datetime_label='timestamp', timestamp_format='Unix', sampling_type='mean', csv_seperator=';', csv_skiplines=0, dictionary='<None>'):
    progr_text = text('Starting Parsing..')
    log_parsing_bar = progress(0)
    progress_counter = 0
    df_list = []
    session_state['sampling_rate'] = sampling_rate
    c = 0  # counter for valid number of logs
    tot = 0  # counter for total Number of Logs
    for uploaded_file in files:
        # CREATE DF
        df = create_df(uploaded_file, sampling_rate=sampling_rate, upload_type=upload_type, datetime_label=datetime_label, timestamp_format=timestamp_format, sampling_type=sampling_type, csv_seperator=csv_seperator, csv_skiplines=csv_skiplines, dictionary=dictionary)
        # APPEND TO LIST
        if df is not None:
            if len(list(df.columns)) > 5 and df.shape[0] > 5:  # Basic Criteria to accept Log
                c += 1
                df_list.append(df.copy())
        else:
            print('df is empty')
            pass
        del df
        gc.collect()
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
                succ_msg = error('❌ Error Loading the Dataset: Corrupted Log Files. Refreshing page in 5 seconds.')
                time.sleep(4)
                succ_msg.empty()
                session_state.clear()
                rerun()
            else:
                df = df_list[0].copy()
                del df_list
                df.dropna(axis=1, how='all', inplace=True)
                session_state['logs_loaded'] = '1/1'

        if session_state['data_imputation']:
            df = df.replace('>', '', regex=True)
            df = df.replace('<', '', regex=True)
            df = df.replace(',', '.', regex=True)
            for c in df.columns:
                if api.types.is_string_dtype(df[c].dtype):
                    try:
                        df[c] = df[c].astype("float64")
                    except ValueError:
                        pass

        print('file_reader :: create_data_table :: Dataframe created successfully with', df.shape[0], 'samples.')
        session_state['df'] = df  # SAVE DATAFRAME TO SESSION
        initialize_session_state_vars(df.shape[1], df.shape[0])
    log_parsing_bar.empty()
    progr_text.empty()
    succ_msg = success('Log Parsing Successful!')
    clear_loading_forms()
    succ_msg.empty()
    time.sleep(2)
    rerun()


def create_data_table_from_bookmark(file_path):
    with spinner('Reading Bookmarked File'):
        df = read_parquet(file_path)

    initialize_session_state_vars(df.shape[1], df.shape[0])
    clear_loading_forms()
    success('Bookmark File Loaded to DataTable Successfully!')
    time.sleep(1)

    return df
