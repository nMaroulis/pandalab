import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from statsmodels.tsa.seasonal import STL
from library.overview.navigation import page_header
from sklearn.neighbors import LocalOutlierFactor

import plotly.graph_objects as go
from plotly.express import scatter

page_header("Anomaly Detection")
st.markdown("<h2 style='text-align: center; color: #373737;'>Anomaly Detection Tool</h2>", unsafe_allow_html=True)

if 'df' not in st.session_state:
    st.sidebar.warning('Data Table is Empty')
    st.markdown("<h2 style='text-align: center; color: #787878; margin-top:120px;'>Data Table is Empty</h2>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center;'><a style='text-align: center; color: #787878; margin-top:40px;' href='/' target='_self'>Load Data</a>", unsafe_allow_html=True)
else:

    with st.form('ad'):

        selected_feature = st.selectbox('Choose Feature to Examine', options=st.session_state.df.columns)

        selected_algo = st.selectbox('Select Algorithm', options=['Isolation Forest', 'Arima', 'HDBSCAN Clustering',
                                                                  'Plausibility Check', 'Outlier Detection [IQR]'])
        ad_submit = st.form_submit_button('Submit', disabled=True)
        if ad_submit:
            with st.spinner("Finding Timeseries Anomalies for " + selected_feature + " using " + selected_algo):
                df = st.session_state.df.copy()
                df = df.dropna()

                corrs = []
                labels = []
                for i in df._get_numeric_data().columns:
                    if i != selected_feature:

                        corr = abs(df[i].corr(df[selected_feature]))

                        if len(corrs) > 4:

                            min_pos = corrs.index(min(corrs))

                            if corr > min_pos:
                                corrs[min_pos] = corr
                                labels[min_pos] = i

                        else:
                            corrs.append(corr)
                            labels.append(i)
                # print(labels, corrs)

                labels.insert(0, selected_feature)
                data = df[labels].values.reshape(-1, 1)

                clf = IsolationForest(contamination=0.2)
                # clf = OneClassSVM()
                # clf = LocalOutlierFactor(n_neighbors=21, contamination=0.1)
                clf.fit(data)  # fit_predict

                predictions = clf.predict(data)  # Predict anomalies (1 for normal, -1 for anomalies)

                # print(predictions)
                # print(df.head())
                # print(df.shape[0], len(predictions))

                p1 = []
                for i in range(len(predictions)):
                    if i % 6 == 0:
                        p1.append(predictions[i])

                df["Anomaly"] = p1  # predictions  # Add a new column to the DataFrame to store anomaly labels

                df.loc[df["Anomaly"] == -1, "Anomaly"] = "Anomaly"
                df.loc[df["Anomaly"] == 1, "Anomaly"] = "Normal"

                from plotly.express import scatter

                # Create a scatter plot using Plotly, highlighting anomalies in red
                import math
                if df.shape[0] > 60000:
                    reduce_by = math.ceil(df.shape[0] / 60000)
                    df = df.groupby(df.index // reduce_by).first()

                fig = scatter(df, x=df.index, y=selected_feature, color="Anomaly", title='Anomaly Detection Plot',
                              color_discrete_map={"Normal": 'blue', "Anomaly": 'red'}, render_mode='webgl')
                fig.update_traces(marker=dict(size=6), selector=dict(mode='markers'), hoverinfo='skip')

                # Show the plot
                st.plotly_chart(fig, use_container_width=True)

    st.warning('⚠️ Currently Unavailable.')
