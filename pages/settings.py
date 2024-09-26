import streamlit as st
from library.file_reader.file_load_form import get_bookmark_archives
from library.overview.navigation import page_header
from datetime import datetime
from settings.settings import VERSION_NUMBER, TRAINING_CACHE_DIR

page_header("Settings")
st.markdown("<h2 style='text-align: center; color: #373737;'>Settings</h2>", unsafe_allow_html=True)

st.sidebar.info('⚙️ Version **' + VERSION_NUMBER + '**')
