import streamlit as st
from library.preprocessing.filtering import filtering_form
from library.preprocessing.cleaning import cleaning_form
from library.preprocessing.selection import selection_form
from library.preprocessing.outlier_detection import outlier_detection_form
from library.preprocessing.transformation import feature_transformation_form
from library.preprocessing.generation import feature_generation_form
from library.overview.navigation import page_header


# PAGE HEADER
page_header("Preprocessing")
st.markdown("<h2 style='text-align: center; color: #373737;'>Data Preprocessing</h2>", unsafe_allow_html=True)


if 'df' not in st.session_state:
    st.sidebar.warning('Data Table is Empty')
    st.markdown("<h2 style='text-align: center; color: #787878; margin-top:120px;'>Data Table is Empty</h2>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center;'><a style='text-align: center; color: #787878; margin-top:40px;' href='/' target='_self'>Load Data</a>", unsafe_allow_html=True)
else:
    st.sidebar.success('Data Table Loaded', icon=":material/rubric:")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Data Cleaning", "Feature Filtering", "Feature Selection", "Outlier Detection",
                                                  "Feature Transformation", "Feature Generation", "Manual Processing", "Data Augmentation"])
    with tab1:  # Data Cleaning
        st.markdown("<h3 style='text-align: left; color: #373737;'>Data Cleaning</h3>", unsafe_allow_html=True)
        st.write("This Functionality will treat Empty/Null values, provide Plausibility check and more. **Methods that remove Rows**")
        cleaning_form()
    with tab2:  # Feature Filtering
        st.markdown("<h3 style='text-align: left; color: #373737;'>Feature Filtering</h3>", unsafe_allow_html=True)
        st.write("This Functionality will detect and remove Static, Multicolliniarity between independent Features and more. **Methods that remove entire Columns**")
        filtering_form()
    with tab3:  # Feature Selection
        st.markdown("<h3 style='text-align: left; color: #373737;'>Feature Selection</h3>", unsafe_allow_html=True)
        st.write(
            "This Functionality will use the Automated Correlation Pipeline in order to **keep the most correlated Features with the Target**")
        selection_form()
    with tab4:  # Outlier Detection
        st.markdown("<h3 style='text-align: left; color: #373737;'>Outlier Detection & Removal</h3>", unsafe_allow_html=True)
        st.write(
            "This Functionality will identify Outlier values in the Features and remove them, based on the algorithm.")
        outlier_detection_form()
    with tab5:  # Feature Transformation
        st.markdown("<h3 style='text-align: left; color: #373737;'>Feature Transformation</h3>", unsafe_allow_html=True)
        st.write("This Functionality will transform the actual Value of each Feature, based on Transformation Techniques.")
        feature_transformation_form()
    with tab6:  # Data Generation
        st.markdown("<h3 style='text-align: left; color: #373737;'>Feature Generation</h3>", unsafe_allow_html=True)
        feature_generation_form()
    with tab7:  # Manual
        st.markdown("<h3 style='text-align: left; color: #373737;'>Manual Processing</h3>", unsafe_allow_html=True)
        with st.form("manual_process"):

            st.markdown("<h5 style='text-align: left; color: #787878;padding:0'>Plausibility Threshold</h5>",
                     unsafe_allow_html=True)
            st.caption("Choose to remove/replace Implausible Values of Features.")
            st.markdown("<hr style='text-align: left; width:5em; margin: 0; color: #5a5b5e'></hr>",
                        unsafe_allow_html=True)

            mp_cols = st.columns([4, 1, 1])
            with mp_cols[0]:
                mp_col = st.selectbox('Feature(s)', options=list(st.session_state.df.columns))
                st.radio('Method', options=['Remove Sample', 'Replace with Null'], horizontal=True)
            with mp_cols[1]:
                mp_min_val = st.number_input('Minimum Value', value=0.0)
            with mp_cols[2]:
                mp_max_val = st.number_input('Maximum Value', value=10.0)

            st.divider()

            submitted_mp = st.form_submit_button("Update Data")
            if submitted_mp:
                with st.spinner('Filtering'):

                    # print(mp_col, mp_min_val, mp_max_val)
                    st.session_state.df = st.session_state.df.drop(st.session_state.df[st.session_state.df[mp_col] < mp_min_val].index)
                    st.session_state.df = st.session_state.df.drop(st.session_state.df[st.session_state.df[mp_col] > mp_max_val].index)

                    # st.session_state.df = st.session_state.df.drop(st.session_state.df[st.session_state.df[mp_col] > mp_max_val].index)
                    st.cache_data.clear()  # CLEAR CACHE

                    st.success('DataTable Cleaned Successfully, based on chosen methods.')
    with tab8:  #  Data Augmentation
        st.warning('Currently not supported.')
        pass
