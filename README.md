<div align="center">
[//]: # (<img src="https://raw.githubusercontent.com/nMaroulis/pandalab/refs/heads/main/static/logo.png">)
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

Welcome to **pandaLab**! This application is your centralized hub for all things crypto. With Sibyl, you can connect multiple crypto exchange accounts, deploy smart trading strategies, and access a wide range of AI-powered tools—all within a secure, locally deployed environment.

[//]: # (<hr>)

### Instructions

#### Dockerfile - Linux

$ sudo su

$ docker build -t pandalab_image .

$ docker run --name pandalab --rm -d --network host pandalab_image

$ docker ps

$ docker exec -it pandalab /bin/bash
