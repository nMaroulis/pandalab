from streamlit import toggle, columns, slider, write, expander, selectbox, number_input, info, caption, latex, code, tabs, html
from pandas import DataFrame
from extra_streamlit_components import stepper_bar


def get_automated_params(model_choise):
    if model_choise == 'Recurrent Neural Network - LSTM [Auto]':
        default_params = {'layers': [100, 64, 32, 0, 0, 0], 'layers_t': [1, 0, 0, 0, 0, 0],
                          'layers_drp': [0.2, 0.2, 0.0, 0.0, 0.0, 0.0], 'n_timesteps': 2, 'l2_reg': 0.0001,
                          'num_epochs': 40, 'batch_size': 32, 'early_stopping': 6}
    elif model_choise == 'Feed-Forward Neural Network [Auto]':
        default_params = {'layers': [64, 32, 32, 0, 0, 0], 'layers_t': [0, 0, 0, 0, 0, 0],
                          'layers_drp': [0.2, 0.0, 0.2, 0.0, 0.0, 0.0], 'n_timesteps': 1, 'l2_reg': 0.0001,
                          'num_epochs': 40, 'batch_size': 32, 'early_stopping': 6}
    else:
        default_params = {'layers': [64, 32, 32, 0, 0, 0], 'layers_t': [0, 0, 0, 0, 0, 0],
                          'layers_drp': [0.2, 0.0, 0.2, 0.0, 0.0, 0.0], 'n_timesteps': 1, 'l2_reg': 0.0001,
                          'num_epochs': 40, 'batch_size': 32, 'early_stopping': 6}

    return default_params


def get_automated_params_xgboost():

    # from numpy import sqrt as np_sqrt, log2 as np_log2
    # n_estimators = int(np_sqrt(session_state.df.shape[0]))
    # min_child_weight = int(np_log2(session_state.df.shape[1])) + 1
    default_params = {"n_estimators": 420, "eta": 0.018, "max_depth": 10, "min_child_weight": 50, "min_split_loss": 0.0,
                   "subsample": 1.0}

    return default_params


def hyperparameter_form(m):
    if m == "KNN":
        with expander("KNN Model Hyperparameters", expanded=True):
            col1, col2, col3 = columns([1, 1, 1])
            with col1:
                n_neighbors = number_input(label="Number of Neighbors (n_neighbors)", min_value=1, max_value=10001, value=11)
                weights = selectbox(label="Weights", options=['uniform'])
            with col2:
                algo = selectbox(label="Algorithm", options=['auto', 'ball_tree', 'kd_tree', 'brute'])
                leaf_size = number_input(label="Leaf Size", min_value=2, max_value=1000, value=30,
                                         help="Leaf size passed to BallTree or KDTree. This can affect the speed of the "
                                              "construction and query, as well as the memory required to store the tree. "
                                              "The optimal value depends on the nature of the problem.")
            with col3:
                metric = selectbox(label="Metric", options=['minkowski'])
                power_param = number_input(label="Power parameter", min_value=1, max_value=10, value=2,
                                           help="Power parameter for the Minkowski metric. When p = 1, this is "
                                                "equivalent to using manhattan_distance (l1), and euclidean_distance "
                                                "(l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.")
        return {"n_neighbors": n_neighbors, "weights": weights, "algorithm": algo, "leaf_size": leaf_size, "p": power_param, "metric": metric}
    elif m == 'XGBoost [Auto]':
        default_params = get_automated_params_xgboost()
        with expander("XGBoost Model Hyperparameters", expanded=True):
            xgboost_estimators = slider("Number of Estimators for the XGBoost Model (default: 300)", 1, 3000, default_params.get('n_estimators'))
            html(f"""
                      ***Note***: Higher Number of estimators means higher accuracy but will substantially increase computation time
                  """)
            col1, col2, col3 = columns([1, 1, 1])
            with col1:
                eta = number_input(label="Learning Rate", min_value=0.0001, max_value=1.0, value=default_params.get('eta'), help="The lower the Learning rate, it is more possible to find the global minima, but also the slower the model training gets.")
            with col2:
                max_depth = number_input(label="Max Depth per Tree", min_value=1, max_value=40, value=default_params.get('max_depth'))
            with col3:
                min_child_weight = number_input(label="Min Child Weight", min_value=0, max_value=100, value=default_params.get('min_child_weight'))
            col4, col5 = columns(2)
            with col4:
                gamma = number_input(label="Gamma", min_value=0.0, max_value=10.0, value=default_params.get('min_split_loss'))
            with col5:
                subsample = number_input(label="Subsample", min_value=0.1, max_value=1.0, value=default_params.get('subsample'), help="Use a subsample of all the Data during the Training to save Time.")
            return {"n_estimators": xgboost_estimators, "eta": eta, "max_depth": max_depth, "min_child_weight": min_child_weight, "min_split_loss": gamma, "subsample": subsample}
    elif m == 'Support Vector Regression (SVR)':
        info("💡 The SVR algorithm requires that the Features are Scaled, so a **Standard Scaling** will be applied automatically before training.")
        with expander("SVR Model Hyperparameters", expanded=True):
            col1, col2, col3 = columns(3)
            with col1:
                svr_kernel = selectbox("Kernel", options=['rbf', 'linear', 'poly', 'sigmoid'],
                                       help="Specifies the kernel type to be used in the algorithm. If none is given, ‘rbf’ will be used. If a callable is given it is used to precompute the kernel matrix.")
                svr_degree = slider("Polynomial Degree", min_value=1, max_value=6, value=3,
                                    help="Degree of the polynomial kernel function (‘poly’). Must be non-negative. Ignored by all other kernels.")
            with col2:
                svr_c = number_input(label="Regularization parameter: C", min_value=0.0, max_value=10.0, value=1.0, help="Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.")
                svr_tol = number_input(label="Tolerance", min_value=0.00001, max_value=1.0, value=0.001, help="Tolerance for stopping criterion.")
            with col3:
                svr_epsilon = number_input(label="Epsilon", min_value=0.0, max_value=10.0, value=1.0, help="It specifies the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value. Must be non-negative.")
            return {"kernel": svr_kernel, "C": svr_c, "epsilon": svr_epsilon, "degree": svr_degree, "tol": svr_tol}
    elif m == 'Feed-Forward Neural Network':
        info("💡 The Neural Network model requires that the Features are Scaled, so a **Standard Scaling** will be applied automatically before training.")
        with expander("Feed-Forward Neural Network Hyperparameters", expanded=True):
            write('Layer Sizes')
            caption("- Max Layer Depth is **5**")
            caption("- If Layer Size is left at 0, it won't be taken into account.")
            col0, col1, col2, col3, col4 = columns(5)
            with col0:
                layer1_ = number_input('Layer 1', min_value=1, max_value=1024, value=64)
            with col1:
                layer2_ = number_input('Layer 2', min_value=0, max_value=1024, value=32)
            with col2:
                layer3_ = number_input('Layer 3', min_value=0, max_value=1024, value=0)
            with col3:
                layer4_ = number_input('Layer 4', min_value=0, max_value=1024, value=0)
            with col4:
                layer5_ = number_input('Layer 5', min_value=0, max_value=1024, value=0)

            write("Training Parameters")
            col5, col6, col7 = columns(3)
            with col5:
                activation_function = selectbox("Activation Function", options=['relu', 'tanh', 'logistic', 'identity'], help="The activation function decides whether a neuron should be activated or not by calculating the weighted sum and further adding bias to it. The purpose of the activation function is to introduce non-linearity into the output of a neuron.")
                alpha = number_input("L2 Reguralization Term", min_value=0.00001, max_value=1.0, value=0.0001, step=1e-5, format="%.4f",
                                    help="Strength of the L2 regularization term. The L2 regularization term is divided by the sample size when added to the loss.")
                early_stopping = toggle('Early Stopping', help="Use early stopping to terminate training when validation score is not improving.", value=True)
            with col6:
                max_iter = number_input(label="Number of Epochs", min_value=1, max_value=1000, value=300, help="Maximum number of iterations. The solver iterates until convergence (determined by ‘tol’) or this number of iterations. For stochastic solvers (‘sgd’, ‘adam’), note that this determines the number of epochs (how many times each data point will be used), not the number of gradient steps..")
                learning_rate_init = number_input(label="Initial Learning Rate", min_value=0.0, max_value=1.0, value=0.003, step=1e-5, format="%.4f",
                                                  help="The initial learning rate used. It controls the step-size in updating the weights. Only used when solver=’sgd’ or ‘adam’.")
                shuffle_samples = toggle('Shuffle Samples', value=False)

            with col7:
                solver_mlp = selectbox("Solver", options=['adam', 'sgd', 'lbfgs'], help="The solver for weight optimization.")

            layers = []
            if layer1_ >= 1:
                layers.append(layer1_)
                if layer2_ >= 1:
                    layers.append(layer2_)
                    if layer3_ >= 1:
                        layers.append(layer3_)
                        if layer4_ >= 1:
                            layers.append(layer4_)
                            if layer5_ >= 1:
                                layers.append(layer5_)
            return {"hidden_layer_sizes": layers, "activation": activation_function, "alpha": alpha,
                    "max_iter": max_iter, "learning_rate_init": learning_rate_init, "solver": solver_mlp,
                    "early_stopping": early_stopping, "shuffle": shuffle_samples, "verbose": True}
    elif m == 'Multinomial Linear Regression':
        with expander("Linear Regression Model Hyperparameters", expanded=True):
            col1, col2 = columns(2)
            with col1:
                lr_degree = slider(label="Polynomial Degree", min_value=1, max_value=5, value=1)
                lr_intercept = toggle(label="Include Intercept", value=True)
                lr_norm_coeffs = toggle(label="Normalize Coefficients (Ridge Regression)", value=False)
            with col2:
                lr_alpha = number_input(label="Ridge Regression Alpha (L2 Penalty)", min_value=0.0, max_value=10.0, value=1.0,
                                         help="Constant that multiplies the L2 term, controlling regularization strength. When alpha = 0, the objective is equivalent to ordinary least squares, solved by the LinearRegression object.")
            info("🛈 Based on the Parameters a **Linear**, **Ridge** or **Lasso** Regression Model will be trained.")

        return {"degree": lr_degree, "intercept": lr_intercept, "norm_coeffs": lr_norm_coeffs, "alpha": lr_alpha}
    else:  # DNN
        default_params = get_automated_params(m)
        info("💡 It is suggested that the Features are **scaled** for this Model. Use MinMax or Standard Scaling above.")
        with expander("Deep Neural Network Hyperparameters", expanded=True):
            write('**1. Model Architecture Parameters**')
            write('Layer Sizes')
            caption("- Max Layer Depth is **6**")
            caption("- If Layer Size is left at 0, it won't be taken into account.")
            col0, col1, col2, col3, col4, col5 = columns(6)
            with col0:
                write('Layer **1**')
                layer1_ = number_input('Layer 1 size', min_value=1, max_value=1024, value=default_params.get('layers')[0])
                layer1_type_ = selectbox('Layer 1 type', options=['Dense', 'LSTM', 'Convolutional'], index=default_params.get('layers_t')[0])
                layer1_act_ = selectbox('Layer 1 Activation Function', options=['relu', 'tanh', 'logistic', 'identity'], help="The activation function decides whether a neuron should be activated or not by calculating the weighted sum and further adding bias to it. The purpose of the activation function is to introduce non-linearity into the output of a neuron.")
                layer1_drop_ = number_input('Layer 1 Dropout', value=default_params.get('layers_drp')[0], min_value=0.0, max_value=0.9, help="Dropout connections between layers to help model generalize.")
            with col1:
                write('Layer **2**')
                layer2_ = number_input('Layer 2 size', min_value=0, max_value=1024, value=default_params.get('layers')[1])
                layer2_type_ = selectbox('Layer 2 type', options=['Dense', 'LSTM', 'Convolutional'], index=default_params.get('layers_t')[1])
                layer2_act_ = selectbox('Layer 2 Activation Function', options=['relu', 'tanh', 'logistic', 'identity'])
                layer2_drop_ = number_input('Layer 2 Dropout', value=default_params.get('layers_drp')[1], min_value=0.0, max_value=0.9)
            with col2:
                write('Layer **3**')
                layer3_ = number_input('Layer 3 size', min_value=0, max_value=1024, value=default_params.get('layers')[2])
                layer3_type_ = selectbox('Layer 3 type', options=['Dense', 'LSTM', 'Convolutional'], index=default_params.get('layers_t')[2])
                layer3_act_ = selectbox('Layer 3 Activation Function', options=['relu', 'tanh', 'logistic', 'identity'])
                layer3_drop_ = number_input('Layer 3 Dropout', value=default_params.get('layers_drp')[2], min_value=0.0, max_value=0.9)
            with col3:
                write('Layer **4**')
                layer4_ = number_input('Layer 4 size', min_value=0, max_value=1024, value=default_params.get('layers')[3])
                layer4_type_ = selectbox('Layer 4 type', options=['Dense', 'LSTM', 'Convolutional'], index=default_params.get('layers_t')[3])
                layer4_act_ = selectbox('Layer 4 Activation Function', options=['relu', 'tanh', 'logistic', 'identity'])
                layer4_drop_ = number_input('Layer 4 Dropout', value=default_params.get('layers_drp')[3], min_value=0.0, max_value=0.9)
            with col4:
                write('Layer **5**')
                layer5_ = number_input('Layer 5 size', min_value=0, max_value=1024, value=default_params.get('layers')[4])
                layer5_type_ = selectbox('Layer 5 type', options=['Dense', 'LSTM', 'Convolutional'], index=default_params.get('layers_t')[4])
                layer5_act_ = selectbox('Layer 5 Activation Function', options=['relu', 'tanh', 'logistic', 'identity'])
                layer5_drop_ = number_input('Layer 5 Dropout', value=default_params.get('layers_drp')[4], min_value=0.0, max_value=0.9)
            with col5:
                write('Layer **6**')
                layer6_ = number_input('Layer 6 size', min_value=0, max_value=1024, value=default_params.get('layers')[5])
                layer6_type_ = selectbox('Layer 6 type', options=['Dense', 'LSTM', 'Convolutional'], index=default_params.get('layers_t')[5])
                layer6_act_ = selectbox('Layer 6 Activation Function', options=['relu', 'tanh', 'logistic', 'identity'])
                layer6_drop_ = number_input('Layer 6 Dropout', value=default_params.get('layers_drp')[5], min_value=0.0, max_value=0.9)

            col01, col11 = columns(2)
            with col01:
                caption('The **LSTM** is a **Recurrent** Neural Network which means it is mostly suitable for **Timeseries Data**, where the sequence and **Temporal aspect** matters.')
                n_timesteps = number_input('Number of Past Timesteps (**only** for **LSTM**)', min_value=1, value=default_params.get('n_timesteps'))
                l2_reg = number_input("L2 Reguralization Term", min_value=0.00001, max_value=1.0, value=default_params.get('l2_reg'),
                                     step=1e-5, format="%.4f",
                                     help="Strength of the L2 regularization term. The L2 regularization term is divided by the sample size when added to the loss.")

            write("**2. Training Parameters**")
            col6, col7, col8 = columns(3)
            with col6:
                num_epochs = number_input(label="Number of Epochs", min_value=1, max_value=1000, value=default_params.get('num_epochs'), help="Maximum number of iterations. The solver iterates until convergence (determined by ‘tol’) or this number of iterations. For stochastic solvers (‘sgd’, ‘adam’), note that this determines the number of epochs (how many times each data point will be used), not the number of gradient steps..")
                batch_size = number_input(label="Batch Size", min_value=1, max_value=2048, value=default_params.get('batch_size'), help="")
                toggle('Shuffle Samples', value=False, disabled=True, help="Shuffle will mess the Results of the Timeseries")
            with col7:
                loss_function = selectbox("Loss Function", options=['Mean Squared Error', 'Mean Absolute Error', 'Mean Absolute Percentage Error', 'Mean Squared Logarithmic Error'])
                early_stopping = number_input('Early Stopping (disabled if 0)', help="Use early stopping to terminate training when validation score is not improving.", min_value=0, value=default_params.get('early_stopping'), max_value=1000)
            with col8:
                optimizer = selectbox("Optimizer", options=['adam', 'sgd', 'lbfgs'], help="The solver for weight optimization.")
                learning_rate_init = number_input(label="Initial Learning Rate", min_value=0.0, max_value=1.0, value=0.003, step=1e-5, format="%.4f",
                                                  help="The initial learning rate used. It controls the step-size in updating the weights. Only used when solver=’sgd’ or ‘adam’.")

            layers, layers_t, layers_act, layers_drp = [], [], [], []
            if layer1_ >= 1:
                layers.append(layer1_)
                layers_t.append(layer1_type_)
                layers_act.append(layer1_act_)
                layers_drp.append(layer1_drop_)
                if layer2_ >= 1:
                    layers.append(layer2_)
                    layers_t.append(layer2_type_)
                    layers_act.append(layer2_act_)
                    layers_drp.append(layer2_drop_)
                    if layer3_ >= 1:
                        layers.append(layer3_)
                        layers_t.append(layer3_type_)
                        layers_act.append(layer3_act_)
                        layers_drp.append(layer3_drop_)
                        if layer4_ >= 1:
                            layers.append(layer4_)
                            layers_t.append(layer4_type_)
                            layers_act.append(layer4_act_)
                            layers_drp.append(layer4_drop_)
                            if layer5_ >= 1:
                                layers.append(layer5_)
                                layers_t.append(layer5_type_)
                                layers_act.append(layer5_act_)
                                layers_drp.append(layer5_drop_)
                                if layer6_ >= 1:
                                    layers.append(layer6_)
                                    layers_t.append(layer6_type_)
                                    layers_act.append(layer6_act_)
                                    layers_drp.append(layer6_drop_)

        return {"layers": layers, "layers_t": layers_t, 'layers_activation_func': layers_act, 'layers_dropout': layers_drp, "n_timesteps": n_timesteps,
                'l2_reg': l2_reg, 'num_epochs': num_epochs, 'batch_size': batch_size, 'loss_function': loss_function,
                'optimizer': optimizer, 'early_stopping': early_stopping, 'learning_rate_init': learning_rate_init}


def get_form_explanation():

    html("<h5 style='text-align: left; color: #787878;padding-bottom:0;'>Virtual Sensor Pipeline Information</h5>")
    caption('Click below to **expand** or **minimize** instructions.')
    with expander('ML Model Building Pipeline Information', expanded=True):
        dec_val = stepper_bar(steps=["Dataset Definition", "Model Selection & Parametrization", "Training & Evaluation", "Export Model"], lock_sequence=False)

        if dec_val == 0:
            html("<h5 style='color: #453f3d'>1.1 &ensp;Training & Test Set</h5><p>Define how many samples of the whole DataTable will be used to: train the Machine Learning Model (train set)</p>")
            write('''
    
            >> **[Train Set]** &ensp; **Train** the Machine Learning Model with this set.
    
            >> **[Test Set]** &ensp; **Test** and **evaluate** the Machine Learning Model. The Test set is never seen by the Model and is used as input in order to predict the *unseen* output (*y_pred*) and compare it with the actual (*y*) in order to evaluate the model.
           
            >> **[Validation Set]** &ensp; While the Test Set is used after the model is trained in order to evaluate it, the validation set is used during the training as unseen data, in order to iteratively validate the model.
    
            ''')
            html("<h5 style='color: #453f3d'>1.2 &ensp; Define Inputs & Outputs</h5><p>Define the Features that constitude the Model's <strong>Input Vector</strong> (independent variables) and the <strong>Output Feature</strong> of the Model (dependent variable).</p>")
            html("<h5 style='color: #453f3d'>1.3 &ensp; Data Transformation</h5><p>Some models (e.g. a Neural Network), require the Input/Output Features to be scaled in order to perform optimally.</p>")
            write('''
    
            >> The **MinMax** scales all the data in the range **[0, 1]** where 0 -> min and 1 -> max.
    
            >> The **Standard Scaling** standardizes features by **removing the mean** and **scaling to unit variance**.
        
            ''')

        if dec_val == 1:

            html("<h5 style='color: #453f3d'>2.1 Model Selection</h5><p>The Models with the suffix <strong>[Auto]</strong> will be initialized automatically by the System.</p>")
            mcol0, mcol1 = columns(2, gap='large')
            with mcol0:
                write("**Low:**")
                write('''
                - XGBoost [Auto]
                - Polynomial Linear Regression
                - Multilayer Perceptron - Neural Network
                - KNN
                - Support Vector Regression (SVR)
                ''')
            with mcol1:
                write("**Deep Neural Networks:**")
                write("+ Recurrent Neural Network - LSTM [Auto]")
                write("+ Feed-Forward Neural Network [Auto]")
                write("+ Convolutional Neural Network [Auto]")

            html("<h5 style='color: #453f3d'>2.2 Model Hyperparameters</h5><p>The User can define the Hyperparameters of the each Model manually. In case the suffix <strong>[Auto]</strong> is after the Model name, the system will configure the appropriate Hyperparameters automatically, but still the User is able to manually edit any parameter.</p>")
            html("<p>The automated parameters are based on the size of the DataTable (rows, columns) and some basic statistics on the Data. In order to have an actual optimized Model the **Hyperparameter Optimization** option has to be chosen.</p>")

        if dec_val == 2:
            write("After the Training of the Model has finished, the Evaluation Results will be generated in this page.")
            html("<h5 style='color: #453f3d'>1. Model Error Scores</h5><p>After the Model has finished training on "
                     "the Training Set, the Test Set will be used in order to evaluate the model. The Inputs of the "
                     "Testing set are given to the Model as Inputs and a **prediction** (y_pred) is generated from the "
                     "model for each Sample of the Test Set. Then in order to evaluate the Prediction capabilities of "
                     "the Model, the Actual Output (y_actual) of the Test Set is compared with the Prediction (y_pred) "
                     "and different metrics are calculated based on that comparison.</p>")
            info('The Results from the Model are based on the Test Set, which means the Model has never actually seen this Data during the Training, so it is evaluated on Unseen Data.')
            write('''
            - ***y_actual*** → The **actual** values of the chosen Target from the Test Set,
            - ***y_pred*** → The **predicted** values from the Model for the chosen Target from the Test Set,
            ''')
            # mae_col, rmse_col = columns(2)
            # with mae_col:
            write('- **Mean Absolute Error**: Calculated as the sum of absolute errors divided by the sample size.')
            latex('MAE = \\frac{\sum_{i=1}^{test\_size}| y\_actual_i-y\_pred_i |}{n\_test}')
            # with rmse_col:
            write('- **Mean Squared Error**: Calculated as the root of the sum of squared errors divided by the sample size.')
            latex('RMSE=\\sqrt{\\frac{\sum_{i=1}^{n\_test}(y\_actual_i-y\_pred_i)^2}{n\_test}}')
            write("- **R-squared (R^2)** is a statistical measure that represents the proportion of the variance for a dependent variable/Output (Y) that's explained by the independent variables/Inputs (X) in the Model.")
            write("Based on values of these Metrics, some automated report messages about the Model's performance are indicated to the User.")
            html("<h5 style='color: #453f3d'>2. Residual (Error) Analysis</h5><p>Plots that analyse the Error of the Model are generated.</p>")
            html("<h5 style='color: #453f3d'>3. Actual vs Predicted Line Plot</h5><p>In this straight-forward Line-plot the Actual Output Feature is plotted (blue line) along with the Prediction from the Model (orange line)</p>")
            html("<h5 style='color: #453f3d'>4. Model Specific Insights</h5><p>Some Machine Information can be exctracted form a Trained Machine Learning Model. These Insights are different across Different Models.</p>")
            write('- Given a **Linear Regression Model**: The Regression Formula is extracted.')
            write('- Given an **XGBoost**: The Feature Importance of the Inputs is extracted.')
            write('- Given a **Neural Network**: The Performance during each epoch/iteration of the Training is extracted.')

        if dec_val == 3:
            html("<h6 style='text-align: left; color: #373737; padding: 0; margin: 0.5em 0 1em 0'>Introduction</h6>")
            write("Once the Machine Learning Model is **trained**, there currently 3 options available on how to utilize, export and **download** the Model."
                  "The Export of the Model will be available after Training in the **Settings** page, where the user is able to define the name of the Model and the method.")



            html("<h5 style='text-align: left; color: #373737; padding: 0; margin: 0.5em 0 1em 0'>Export Option 1: ONNX File for MATLAB</h5>")
            write('''
            > The Machine Learning is exported int ONNX Format, which can then imported/loaded into **Matlab**, in order to start using it for inference.
            - Please note that MATLAB requires the installation of the **Deep Learning Toolbox**.
            - The **Deep Learning Toolbox Converter for ONNX Model Format support package** add-on is required in order to import the ONNX Model.
            ''')
            code(
                f"""onnxFilePath = 'path_to_file/model_name.onnx'; \nmodel = importONNXNetwork(onnxFilePath, "InputDataFormats", "BC", "OutputDataFormats","BC"); \n% x = 2D Matrix with the input Data in the right order\ny_pred = predict(model, x); % y_pred now contains a 1D Vector with the Model's Prediction for each Row of the x Matrix\nfprintf(y_pred) % print predicted values""",
                language='matlab', line_numbers=True)
            caption('The ONNX Model can also be imported in Python.')


            html("<h5 style='text-align: left; color: #373737; padding: 0; margin: 0.5em 0 1em 0'>Export Option 2: HDF5 or Pickle File</h5>")
            write("> This option is available only for the Deep neural Network Models. The .h5 file can be imported/loaded into Python and Matlab.")
            write("> This option is available only for the Machine Learning. The .pkl file can easily be imported/loaded into Python. For Matlab it is reuired to use the python tool inside Matlab.")

            write("- Tensorflow **HDF5**")
            write("The **Deep Learning Toolbox Converter for ONNX Model Format support package** add-on is required.")
            code(f"""modelfile ='path_to_model/model.h5';\nnet = importKerasNetwork(modelfile);\nplot net;""",
                    language='matlab', line_numbers=True)
            write("")
            html("<h6 style='text-align: left; color: #373737; padding: 0; margin: 0.5em 0 1em 0'>Python:</h6>")
            write("- Tensorflow **HDF5**")
            code('from tensorflow import keras as kr\nmodel = kr.load("path_to_model/model.h5")\nmodel.predict(TEST_X_INPUTS)',
                 language='python', line_numbers=True)
            write("- **Pickle**")
            code('import pickle as pkl\nmodel = pkl.load(open("path_to_model/model.pkl", "rb"))\nmodel.predict(TEST_X_INPUTS)',
                 language='python', line_numbers=True)
    return
