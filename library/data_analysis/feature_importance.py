from streamlit import slider, columns, write, form, multiselect, checkbox, session_state, number_input, expander, \
    markdown, radio, selectbox, container, form_submit_button, spinner, pyplot, code, error, select_slider, toggle, fragment
from numpy import number as np_number
from library.data_analysis.feature_importance_models import get_xgboost_shap_analysis, get_default_feature_importance, get_xgboost_permutation_importance

def generate_input_list(input_features_list, input_features_all, selected_target):

    if input_features_all:
        input_features_list = session_state.df.select_dtypes(include=np_number).columns.tolist()
        input_features_list.remove(selected_target)
    else:
        if selected_target in input_features_list:  # If target selected in both vectors
            input_features_list.remove(selected_target)
    return input_features_list

@fragment
def get_fi_form():
    with form("fi_form"):
        write("The Feature Importance is calculated by training an ***XGBoost*** Tree-based Machine Learning Model, using "
              "the input Data in order to predict the Target Value. From this Model, **explainableAI** techniques are available, in order to "
              "analyse which factors have the larger impact on the Target value. The more robust solution, the **SHAPley** values, "
              "a feature importance method based on game theory, explain the fraction of the model output variability attributable to each feature by percentage")

        with container():
            write('Define Model Inputs & Output (Target)')
            col1_in, col2_in = columns([3, 1])
            with col1_in:
                input_features_list = multiselect(label="Define Input Features", options=list(session_state.df._get_numeric_data().columns))
                input_features_all = checkbox(label="Use all Features", value=False)
            with col2_in:
                # Choose Target for Training
                target_options = list(session_state.df._get_numeric_data().columns)
                selected_target = selectbox(label='Select Target',
                                            options=target_options,
                                            help="The model will be created to provide Feature Importance for the specified Target.")

        # slowest_time = str( round(((0.0003 * session_state.df.shape[0]) / 60), 1) )
        automatic_model_speed = select_slider("Select ML Model Training Speed. (The more time it takes, the result becomes more Robust.)",
                                    ['Express [30s]', 'Fast [1.5m]', 'Normal [3m]', 'Slow [6m]', 'All Data [~12m]'], 'Normal [3m]')

        # write("ML Model Training for Feature Importance. Please choose Preferred Options:")
        with expander("Advanced Model Hyperparameters", expanded=False):
            xgboost_estimators = slider("Number of Estimators for the XGBoost Model (default: 300)", 1, 3000, 300)
            markdown(f"""
                      ***Note***: Higher Number of estimators means higher accuracy but will substantially increase computation time
                  """)
            col1, col2, col3 = columns([1, 1, 1])
            with col1:
                eta = number_input(label="Learning Rate", min_value=0.0001, max_value=1.0, value=0.018)
            with col2:
                max_depth = number_input(label="Max Depth per Tree", min_value=1, max_value=40, value=10)
            with col3:
                min_child_weight = number_input(label="Min Child Weight", min_value=0, max_value=20, value=18)
            col4, col5 = columns(2)
            with col4:
                gamma = number_input(label="Gamma", min_value=0.0, max_value=10.0, value=0.0)
            with col5:
                subsample = number_input(label="Subsample", min_value=0.1, max_value=1.0, value=1.0)

        with expander("SHAP Analysis Parameters", expanded=False):
            write("Plots:")
            fi_cols = columns(5)
            shap_plots = [0, 0, 0, 0, 0]
            with fi_cols[0]:
                shap_plots[0] = checkbox("(1) Global Feature Importance", value=True)
            with fi_cols[1]:
                shap_plots[1] = checkbox("(2) Summary Plot", value=True)
            with fi_cols[2]:
                shap_plots[2] = checkbox("(3) Dependence Plot", value=True)
            with fi_cols[3]:
                shap_plots[3] = checkbox("(4) Dependence Plot (3-var)", value=False)
            with fi_cols[4]:
                shap_plots[4] = checkbox("(5) Heatmap", value=False)

            cluster_num = select_slider("(1.1) Number of Automatic Clusters for Global Feature Importance", ['None', '2', '3', '4'],'None') # 'Scatter Plot', 'Dependence Plot', 'Summary Plot',
            markdown(f"""***Note***: Based on the number of clusters chosen, the Feature Importance will be partitioned 
            into that many different bars""")
        markdown("<hr style='text-align: left; width:8em; margin: 1em 0em 1em; color: #5a5b5e'></hr>", unsafe_allow_html=True)

        # TRAINING/TEST TRAIN-PLOT SET CHOICE
        default_training_set = True
        if session_state.df.shape[0] > 2000:
            default_training_set = False
        use_train_set_for_all = toggle('Use all Data for Training', value=default_training_set, help="In case of Large Datasets, Disable this option")

        # TYPE OF IMPORTANCE
        type_of_importance = radio('Select Type of importance',
                                      options=['Shapley Values [Slow]', 'Information Gain', 'Permutation Importance'], index=0,
                                      horizontal=True, help='Permutation Importance: The ML Model is trained and '
                                                            'then each single feature is shuffled, in order to see '
                                                            'which affects the Model prediction more. The more the '
                                                            'prediction is affected, the more important it is considered.')
        # Estimated Training Time Formula - rows/1000 * estimators/4000 / learning_rate
        ett = str(int(((session_state.df.shape[0] / 1000) * (xgboost_estimators / 4000)) / eta))
        code("🕑 Maximum Estimated Execution Time " + ett + " seconds, for " + str(
            session_state.df.shape[0]) + " training samples.", language=None)
        # Every form must have a submit button.
        submitted = form_submit_button("Train Model")
        if submitted:
            # GET FEATURES FOR INPUTS/OUTPUT
            input_features_list = generate_input_list(input_features_list, input_features_all, selected_target)

            if len(input_features_list) < 1:
                error("Input Feature List is Empty! Please choose some Inputs.")
                return

            if type_of_importance == 'Information Gain':
                with spinner("Training XGBoost ML Model to obtain Feature Importance..."):
                    fi_plot = get_default_feature_importance(session_state.df, xgboost_estimators, eta, max_depth, min_child_weight,
                                          gamma, subsample, selected_target, input_features_list)
                    pyplot(fi_plot)
            elif type_of_importance == 'Permutation Importance':  # Permutation Importance
                write("The permutation feature importance is defined to be the decrease in a ML model score when a single feature value is ***randomly shuffled***. This procedure breaks the relationship between the feature and the target, thus the drop in the model score is indicative of how much the model depends on the feature.")
                with spinner("Training XGBoost ML Model to obtain Feature Permutation Importance..."):
                    fi_plot = get_xgboost_permutation_importance(session_state.df, xgboost_estimators, eta, max_depth,
                                                      min_child_weight,
                                                      gamma, subsample, selected_target, input_features_list)
                    pyplot(fi_plot)
            else:  # SHAP VALUES
                get_xgboost_shap_analysis(input_features_list, selected_target, automatic_model_speed, shap_plots,
                                          xgboost_estimators, eta, max_depth, min_child_weight, gamma, subsample,
                                          cluster_num, use_train_set_for_all)
    return
