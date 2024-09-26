from streamlit import (subheader, table, markdown, toggle, columns, slider, write, container, button, spinner,
                       success, expander, selectbox, session_state, multiselect)
from pandas import DataFrame


def classification_form():
    subheader("Classification Modelling Parameters")
    # with st.form("training_dataset"):
    with expander("Training Dataset Parameters", expanded=True):
        df = DataFrame(
            [["Training Set Definition", "Define Train/Test/Validation set sizes", "Yes"],
             ["Model Selection", "Choose Classification Model", "Yes"],
             ["Model Hyperparameters", "Choose Hyperparameters for the Model", "Yes"],
             ["Automated Hyperparameter Optimization", "Choose Optimization Technique", "No"],
             ["Evaluation Metric", "Choose Evaluation Metric to be displayed after Training", "No"],
             ], columns=["Parameter", "Description", "Mandatory"])
        table(df)

    markdown("<h5 style='text-align: left; color: #787878;'>Training Set Parameters</h5>", unsafe_allow_html=True)
    markdown("<hr style='text-align: left; width:8em; margin: 0; color: #5a5b5e'></hr>", unsafe_allow_html=True)
    col01, col02 = columns(2)
    with col01:
        training_set_size = slider("Percentage (%) of the DataTable Samples to be Used for Training", 1, 100, 80)
        toggle('Shuffle Data')
    with col02:
        test_set_size = slider("Percentage (%) of the Test Set to be Used for Validation", 1, 100, 20)

    with container():
        write('Define Model Inputs & Output (Target)')
        col1_in, col2_in = columns([3, 1])
        with col1_in:
            input_features_list = multiselect(label="Define Input Features",
                                                 options=list(session_state.df.columns))
            input_features_all = toggle(label="Use all Features", value=False)
        with col2_in:
            # Choose Target for Training
            target_options = list(session_state.df.columns)
            selected_target = selectbox(label='Select Target', options=target_options,
                                           help="The model will be created to provide Feature Importance for the specified Target.")

    markdown("<h5 style='text-align: left; color: #787878;'>Model Selection</h5>", unsafe_allow_html=True)
    markdown("<hr style='text-align: left; width:8em; margin: 0; color: #5a5b5e'></hr>", unsafe_allow_html=True)
    selectbox('Choose Machine Learning Model', options=['KNN', 'Multinomial Linear Regression', 'SVM'])

    # st.code("🕑 Estimated Execution Time " + str(ceil(log(st.session_state.df.shape[0]))) + " seconds, for " + str(st.session_state.df.shape[0]) + " samples.", language=None)
    cl_submitted = button("Start Training")
    if cl_submitted:
        with spinner("Training Model"):
            import time
            time.sleep(3)
            success('Accuracy 100%')
