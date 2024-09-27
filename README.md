<div align="center">
<img src="https://raw.githubusercontent.com/nMaroulis/pandalab/refs/heads/main/static/logo.png">
</div>

[![Python](https://img.shields.io/badge/python-v3.12-yellow)]()
[![Streamlit](https://img.shields.io/badge/streamlit-v1.38-red)]()
[![Tensorflow](https://img.shields.io/badge/tensorflow-v2.18-orange)]()
[![Pandas](https://img.shields.io/badge/pandas-v2.2.2-blue)]()
[![Plotly](https://img.shields.io/badge/plotly-v5.19-green)]()
[![Sklearn](https://img.shields.io/badge/Scikit_Learn-v1.4.2-purple)]()


# pandaLab - make ML processes a piece of cake!!

[//]: # (<hr>)

[//]: # (<span style="color: red; font-size: 16px;">pre-alpha version</span>)

[//]: # (<br>)

Welcome to **pandaLab**! PandaLab is a powerful yet user-friendly Streamlit application for data exploration and machine learning. Users can upload one or multiple datasets (CSV, Parquet, Excel, etc.), which are automatically combined into a single pandas DataFrame stored in memory. The app provides a suite of tools for data preprocessing, in-depth data analysis, anomaly detection and a ML Model building module, where you can choose from a plethora of ML Models (DNNs, LSTMS, XGboost and many more..), fine-tune, train, evaluate and export the model.
Additionally, PandaLab integrates an OpenAI API that allows you to chat with the DataFrame for more intuitive exploration. Whether you're analyzing patterns, building models, or detecting anomalies, PandaLab is your friendly workspace for data experimentation.

[//]: # (<hr>)

### Instructions

#### Dockerfile - Linux

$ sudo su

$ docker build -t pandalab_image .

$ docker run --name pandalab --rm -d --network host pandalab_image

$ docker ps

$ docker exec -it pandalab /bin/bash
