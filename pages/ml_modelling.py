import streamlit as st
from library.overview.navigation import page_header
from library.modelling.regression_form import regression_form
from library.modelling.classification_form import classification_form
from library.modelling.training.training_main import check_training_status, training_main
from library.modelling.training.training_db_handler import clear_db
from database.db_client import get_training_progress
import time


# PAGE HEADER
page_header("ML Model Builder")
st.markdown("<h2 style='text-align: center; color: #373737;'>Machine Learning Model Builder</h2>", unsafe_allow_html=True)

# SIDEBAR
if 'df' not in st.session_state:
    st.sidebar.warning('Data Table is Empty')
    st.markdown("<h2 style='text-align: center; color: #787878; margin-top:120px;'>Data Table is Empty</h2>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center;'><a style='text-align: center; color: #787878; margin-top:40px;' href='/' target='_self'>Load Data</a>", unsafe_allow_html=True)
else:
    st.sidebar.write('DataTable loaded', icon=":material/rubric:")
    if 'training_in_progress' not in st.session_state or st.session_state['training_in_progress'] is False:
        m_type = st.sidebar.selectbox('Type:', options=["Regression [Supervised]", "Classification [Supervised]", "Clustering [Unsupervised]"])

        if m_type == "Regression [Supervised]":
            regression_form()
        elif m_type == "Classification [Supervised]":
            classification_form()
        else:
            st.subheader('TBA')
    else:
        if check_training_status():
            st.sidebar.write('✅ Training completed')
            if st.sidebar.button('Reset Model', type='primary'):
                st.session_state['training_in_progress'] = False
                clear_db()
                st.rerun()
            training_main()
        else:
            st.toast('🕒 Training in progress...')
            st.sidebar.write('🕒 Training in progress...')

            if st.sidebar.button('Stop Training', type='primary'):
                st.session_state['training_in_progress'] = False
                clear_db()
                if 'training_pid' in st.session_state:
                    st.session_state['training_pid'].kill()
                st.rerun()

            st.markdown(
                "<h3 style='text-align: center; color: #787878; margin-top:120px;'>Model Training in Progress</h3>",
                unsafe_allow_html=True)
            st.markdown(
                '<p style="text-align:center;"><img src="https://i.gifer.com/ZKZg.gif" alt="Loading" style="width:48px;height:48px;"</p>',
                unsafe_allow_html=True)
            print_status = st.markdown("""""")
            while True:
                message = get_training_progress(st.session_state['training_pid'].pid)
                if message == 'process exit':
                    st.rerun()
                print_status.markdown(f"<div style='text-align:center;'>\
                            <p style='text-align: center; color:#787878;margin-top:5px;'>{message}</p>\
                            </div>", unsafe_allow_html=True)
                time.sleep(2)
            # print(st.session_state['training_pid'].pid)
            # # # st.write(st.session_state['training_pid'].stdout)
            # # # st.write(st.session_state['training_pid'].stdout.readline())
            # for line in iter(st.session_state['training_pid'].stdout.readline, b''):  # b'\n'-separated lines
            #     st.write('got line from subprocess: %r', line)
