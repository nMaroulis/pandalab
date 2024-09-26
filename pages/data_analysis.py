import streamlit as st
from library.data_analysis.regression_analysis import get_regression_analysis_form
from library.data_analysis.correlation_analysis import get_correlation_form
from library.data_analysis.feature_importance import get_fi_form
# from library.navigation import get_date_interval_picker
from library.data_analysis.misc_analysis import get_contour_map_form, get_pairplot_form, get_pcp_form
from library.data_analysis.granger_causality_test import get_causality_form
from library.overview.navigation import page_header

page_header("Data Analysis")
st.html("<h2 style='text-align: center; color: #373737;'>Data Analysis</h2>")


if 'df' not in st.session_state:
    st.sidebar.warning('DataTable is Empty')
    st.html("<h2 style='text-align: center; color: #787878; margin-top:120px;'>Data Table is Empty</h2>")
    st.html("<div style='text-align: center;'><a style='text-align: center; color: #787878; margin-top:40px;' href='/' target='_self'>Load Data</a>")
else:
    st.sidebar.success('DataTable loaded', icon=":material/rubric:")
    st.sidebar.button('PDF Report', disabled=True, use_container_width=True)
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Regression Plot", "Correlation Heatmap", "Feature Importance", "Visualization Tool", "Causality Test",  "Statistical Report"])
    with tab1:  # Reg Plot
        st.subheader("Regression Analysis")
        st.write("The Plot will visualize the Relationship between two or three Features")
        get_regression_analysis_form()
    with tab2:  # Heatmap
        st.subheader("Correlation Heatmap")
        get_correlation_form()
    with tab3:  # Feature Importance
        st.subheader("Feature Importance")
        get_fi_form()
    with tab4:  # Visualization Tools
        st.subheader("Visualisation Tool")
        tab50, tab51, tab52 = st.tabs(['Contour Map', 'Distribution Pairplot', 'Parallel Coordinates Plot'])
        with tab50:  # Contour
            get_contour_map_form()
        with tab51:
            get_pairplot_form()
        with tab52:
            get_pcp_form()
    with tab5:
        st.subheader("Granger Causality Test")
        st.markdown(f""":red[Experimental]""")
        get_causality_form()
    with tab6:
        st.subheader("Autocorrelation Test")
        st.markdown(f""":red[Experimental]""")