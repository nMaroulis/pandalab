import pandas as pd
from streamlit import write, info, form, radio, columns, markdown, expander, form_submit_button, spinner, plotly_chart,\
    table, session_state, selectbox, multiselect, caption, warning, html, fragment
from scipy.spatial.distance import correlation  # from scipy.signal import correlate
from plotly.express import imshow


def correlation_heatmap(formula='pearson', corr_target='', corr_cols=None):

    # IF CHOSEN VARIABLES
    if len(corr_cols) > 1:
        correlations = session_state.df[corr_cols].corr(method=formula, numeric_only=True)
        if corr_target in corr_cols:
            corr_to_ = len(corr_cols) if len(corr_cols) < 11 else 11
            top10_corr = correlations.sort_values(by=corr_target, ascending=False, key=abs)[corr_target][1:corr_to_]
        else:
            top10_corr = None
    # IF ALL COLUMNS
    else:
        correlations = session_state.df._get_numeric_data().corr(method=formula, numeric_only=True)
        corr_to_ = -1 if len(session_state.df._get_numeric_data().columns) < 11 else 11
        top10_corr = correlations.sort_values(by=corr_target, ascending=False, key=abs)[corr_target][1:corr_to_]
    # fig, ax = plt.subplots(figsize=(16, 16))
    if formula == "pearson":
        cmap = "RdBu_r"
    else:
        cmap = "YlGnBu"  # color_palette("coolwarm", 12)
    fig = imshow(correlations, text_auto='.2f', aspect="auto", height=1000, color_continuous_scale=cmap)

    return fig, top10_corr

@fragment
def get_correlation_form():
    write(
        'The Correlation Heatmap indicates the **linear** correlation between each feature using the ***Pearson Correlation***, ***Spearman Correlation*** or ***Kendall Correlation*** Formula.')
    info("💡 The value range is [-1, 1] where 1 indicates a perfect correlation and -1 a perfect negative "
            "correlation. A correlation of -0.3 < p < 0.3 is considered weak.")
    with form("hm_form"):
        corr_method = radio(
            "Choose the Correlation Formula",
            ('pearson', 'spearman', 'kendall', 'distance', 'partial correlation'), horizontal=True)

        caption('If left empty all columns will be used!')
        corr_cols = multiselect('Choose Specific Columns (Optional)', help="If left empty all columns will be used", options=list(session_state.df._get_numeric_data().columns))

        cr_col, _ = columns([1, 2])
        with cr_col:
            corr_target = selectbox('Show **Top 10** Correlations for:', options=list(session_state.df._get_numeric_data().columns))
        with expander('Partial Correlation Control Features'):
            caption('Choose which Features to control. Only in the case of Partial Correlation!!')
            partial_corr_cols = multiselect('Partial Correlation Columns', options=list(session_state.df.columns))
        markdown(f"""***Descriptions***""")
        colhm1, colhm2, colhm3, colhm4 = columns(4)
        with colhm1:
            with expander(label="Pearson's r", expanded=False):
                markdown(f"""
                  The ***Pearson's*** correlation coefficient (r) is a measure of linear correlation between two variables. 
                  It's value lies between -1 and +1, -1 indicating total negative linear correlation, 0 indicating no linear correlation 
                  and 1 indicating total positive linear correlation. Furthermore, r is invariant under separate changes in location and 
                  scale of the two variables, implying that for a linear function the angle to the x-axis does not affect r. 
                  To calculate r for two variables X and Y, one divides the covariance of X and Y by the product of their standard deviations.""")
        with colhm2:
            with expander(label="Spearman's ρ", expanded=False):
                markdown(f"""
                              The ***Spearman's*** rank correlation coefficient (ρ) is a measure of monotonic correlation between two variables, and is 
                              therefore better in catching nonlinear monotonic correlations than Pearson's r. It's value lies between -1 and +1, -1 indicating total 
                              negative monotonic correlation, 0 indicating no monotonic correlation and 1 indicating total positive monotonic correlation.
                              To calculate ρ for two variables X and Y, one divides the covariance of the rank variables of X and Y by the product of their standard deviations.""")
        with colhm3:
            with expander(label="Kendalls's τ", expanded=False):
                markdown(f"""
                              Similarly to Spearman's rank correlation coefficient, the ***Kendall rank*** correlation coefficient (τ) measures ordinal association between two variables. 
                              It's value lies between -1 and +1, -1 indicating total negative correlation, 0 indicating no correlation and 1 indicating total positive correlation. 
                              To calculate τ for two variables X and Y, one determines the number of concordant and discordant pairs of observations. τ is given by the number of concordant pairs 
                              minus the discordant pairs divided by the total number of pairs.""")
        with colhm4:
            with expander(label="Distance Correlation", expanded=False):
                markdown(f"""Distance correlation is a measure of association strength between non-linear random variables. It goes beyond Pearson’s correlation because it can spot more than linear 
                associations and it can work multi-dimensionally. Distance correlation ranges from 0 to 1, where 0 implies independence between X & Y and 1 implies that the
                 linear subspaces of X & Y are equal.""")

        submitted_hm = form_submit_button("Generate Heatmap")
        if submitted_hm:
            with spinner('Generating Heatmap...'):
                if corr_method == 'distance':
                    d = []
                    if len(corr_cols) > 1:
                        d_cols = corr_cols
                    else:
                        d_cols = session_state.df.columns
                    for i in d_cols:
                        d.append([i, correlation(session_state.df[i].values, session_state.df[corr_target].values)])
                    d = sorted(d, key=lambda x: x[1])
                    table(d)
                elif corr_method == 'partial correlation':
                    from pingouin import partial_corr
                    already_checked = []
                    num_cols = list(session_state.df._get_numeric_data().columns)
                    res = [[0]*len(num_cols) for i in range(len(num_cols))]

                    res_tmp = []
                    c = 0
                    for c1 in num_cols:
                        for c2 in num_cols:
                                if c1 != c2 and c1 not in partial_corr_cols and c2 not in partial_corr_cols:
                                    if c2 not in already_checked:
                                        res_tmp.append(None)
                                    else:
                                        res_tmp.append(partial_corr(data=session_state.df.dropna(), x=c1, y=c2, covar=partial_corr_cols, method='pearson').round(3).r[0])
                                else:
                                    res_tmp.append(None)

                        res[c] = res_tmp
                        res_tmp = []
                        c += 1
                        already_checked.append(c1)

                    pdf = pd.DataFrame(res, columns=num_cols)
                    pdf['Feature'] = num_cols
                    pdf = pdf.set_index('Feature', drop=True)
                    fig = imshow(pdf, text_auto='.2f', aspect="auto", height=1000)
                    plotly_chart(fig, use_container_width=True)
                else:
                    hp, top10_corr = correlation_heatmap(corr_method, corr_target, corr_cols)
                    html("<h5 style='color: #525355;'>Heatmap with all Features</h5>")
                    html("<hr style='text-align: left; margin: 0; color: #5a5b5e'></hr>")
                    plotly_chart(hp, use_container_width=True)
                    html("<h5 style='color: #525355;'>Top 10 Feature Correlations with " + corr_target)
                    html("<hr style='text-align: left; margin: 0; color: #5a5b5e'></hr>")
                    if top10_corr is None:
                        warning('The chosen Target was not in the Input Set')
                    else:
                        table(top10_corr)
    return
