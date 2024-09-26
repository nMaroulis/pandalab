from streamlit import session_state, cache_data


def initialize_session_state_vars(num_cols, num_rows):
    session_state['initial features'] = num_cols
    session_state['initial samples'] = num_rows
    session_state['cleaned features'] = 0
    session_state['filtered features'] = [0, 0, 0, 0]  # static, autocorrelated
    session_state['generated features'] = 0
    session_state['low corr features'] = 0
    session_state['navigation_status'] = 'first_load'
    return


def clear_loading_forms():
    session_state['file_uploading_options_container'].empty()
    # session_state['file_loading_options_container'].empty()
    cache_data.clear()  # clear everything previous
    return
