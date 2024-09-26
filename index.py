import streamlit as st

pages = {

	"File Loader": [
		st.Page("pages/data_upload.py", title="Data Uploader", icon=":material/upload:")
	],
	"Data": [
		st.Page("pages/preprocessing.py", title="Preprocessing", icon=":material/edit_note:"),
		st.Page("pages/data_analysis.py", title="Data Analysis", icon=":material/query_stats:"),
		st.Page("pages/chatbot.py", title="Chat with your Data", icon=":material/chat:"),
		st.Page("pages/dictionary.py", title="Dictionaries", icon=":material/library_books:"),
	],
	"Machine Learning Modelling": [
		st.Page("pages/ml_modelling.py", title="Build ML Model", icon=":material/build:"),
		st.Page("pages/anomaly_detection.py", title="Anomaly Detection", icon=":material/search_off:"),
	],
	"Settings": [
		st.Page("pages/settings.py", title="Settings", icon=":material/settings:")
	]
}

pg = st.navigation(pages)
pg.run()
