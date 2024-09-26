from statsmodels.tsa.stattools import grangercausalitytests, adfuller, kpss
from statsmodels.tools.sm_exceptions import InfeasibleTestError
from streamlit import session_state, progress, text, error, form, form_submit_button, write, columns, markdown,\
    expander, code, spinner, header, number_input, selectbox, table
from numpy import min as np_min
from numpy import sqrt as np_sqrt
from pandas import DataFrame, concat


def make_data_stationary(col1, col2):
    # df_log = np_sqrt(session_state.df[c])
    # df_diff = df_log.diff()
    # print(df_diff.shape)
    # df_log = np_sqrt(session_state.df[target])
    # df_diff1 = df_log.diff()
    #
    # df = concat([df_diff, df_diff1], axis=1)
    # df = df.dropna()
    # print(df_diff1.shape)
    # print(df.shape)
    return

def get_granger_causality(maxlag=12, test='ssr_chi2test', verbose=False, target: str = ""):

    variables = list(session_state.df.columns)
    variables.remove(target)

    res = []

    stationary_res = []
    for i in variables:
        # print(i)
        get_kpss_stationarity_score(session_state.df[i].dropna())
        get_adfuller_stationarity_score(session_state.df[i].dropna())
        stationary_res.append([i, get_kpss_stationarity_score(session_state.df[i].dropna()), get_adfuller_stationarity_score(session_state.df[i].dropna())])
    print(stationary_res)
    progr_text = text('Starting Parsing..')
    log_parsing_bar = progress(0)
    progress_counter = 0
    failed_calc = []

    for c in variables:
        progress_counter += 1
        progr_text.text('Calculating Granger Causality for ' + str(c))
        log_parsing_bar.progress(progress_counter / len(variables))
        try:
            test_result = grangercausalitytests(session_state.df[[c,target]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i + 1][0][test][1], 4) for i in range(maxlag)]
            # if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np_min(p_values)
            res.append(min_p_value)
        except InfeasibleTestError:
            print("LOG :: Granger Causality :: InfeasibleTestError for",c)
            failed_calc.append(c)
            continue

    col1 = 'Correlation with ' + target
    for i in failed_calc:
        variables.remove(i)
    if len(failed_calc) > 0:
        error("The following calculations Failed " + str(failed_calc))
    df = DataFrame({"Feature": variables, col1: res})
    df = df.sort_values(by=col1, ascending=True)
    return df


def get_adfuller_stationarity_score(df_col):
    # print(session_state.df.loc[:, session_state.df.columns != 'DateTime'].values[0])
    result = adfuller(df_col.values)
    # print('ADF Statistics: %f' % result[0])
    # print('p-value: %f' % result[1])
    # print('Critical values:')
    # for key, value in result[4].items():
    #     print('\t%s: %.3f' % (key, value))

    return result  # [0]


def get_kpss_stationarity_score(df_col):
    statistic, p_value, n_lags, critical_values = kpss(df_col.values)

    # print(f'KPSS Statistic: {statistic}')
    # print(f'p-value: {p_value}')
    # print(f'num lags: {n_lags}')
    # print('Critial Values:')
    # for key, value in critical_values.items():
    #     print(f'   {key} : {value}')

    return statistic


def get_causality_form():
    with form("gct_form"):
        write(
            "Granger Causality Algorithm in order to obtain the most affecting factors for the chosen Feature. Please choose Preferred Feature:")
        # Choose Target for Training
        target_options_g = list(session_state.df.columns)
        colg = columns(3)
        with colg[0]:
            selected_target_g = selectbox(label='Select Target',
                                             options=target_options_g,
                                             help="The Granger Causality Algorithm will be initiated for the specified Target")
        with expander("Advanced Algorithm Parameters", expanded=False):
            markdown(f"""***Max Lag***: is the Maximum possible Time Delay. """)
            maxlag = number_input(label="Maximum Number of Lags (seconds)", min_value=1, max_value=60, value=12)
        markdown(f"""
                            ***Info***: Granger causality tests whether one variable in a linear relation can be meaningfully
                            described as a dependent variable and the other variable as an independent variable, whether the \
                            relation is bidirectional, or whether no functional relation exists at all.
                            The algorithm checks the ***Granger Causality*** of all possible combinations of the Data Table with the chosen Target.
                            The rows are the response variable, columns are predictors. The values in the table
                            are the P-Values. **P-Values lesser than the significance level (0.05)**, implies
                            the Null Hypothesis that the coefficients of the corresponding past values is
                            zero, that is, the *X does not cause Y* can be rejected.
                        """)
        code(
            "🕑 Estimated Execution Time 30 seconds, for " + str(session_state.df.shape[0]) + " samples.",
            language=None)
        causality_submit = form_submit_button("Run Causality Test", disabled=True)
        if causality_submit:
            with spinner("Running Granger Causality Test Algorithm..."):
                header('Causal Features for ' + str(selected_target_g) + ' in ascending order')
                table(get_granger_causality(target=selected_target_g, maxlag=maxlag))
