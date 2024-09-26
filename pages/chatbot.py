from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st
import os
from library.overview.navigation import page_header

page_header("💬")
st.html("<h2 style='text-align: center; color: #373737;'>Data ChatBot</h2>")
st.html("<h6 style='text-align: center; color: #48494B;'>Data ChatBot: Chat with the DataTable</h6>")


def clear_submit():
    st.session_state["submit"] = False

if 'df' in st.session_state:
    if 'openai_api_key' not in st.session_state:
        st.warning("Please provide your OpenAI API key in order to be able to use this functionality!")
        with st.form('openai_api_key'):
            api_key = st.text_input('OpenAI API key:', placeholder='Enter your OpenAI API key')
            sub = st.form_submit_button('Update Key')
            if sub:
                st.session_state['openai_api_key'] = api_key
                st.rerun()
    else:
        if "cb_messages" not in st.session_state:
            st.session_state["cb_messages"] = [{"role": "assistant", "content": "How can I help you?"}]

        with st.sidebar:
            if st.button("Clear conversation history", type='primary', use_container_width=True):
                st.session_state["cb_messages"] = [{"role": "assistant", "content": "How can I help you?"}]
            if st.button("Clear API Key", use_container_width=True):
                del st.session_state['openai_api_key']
                st.rerun()
            with st.expander('Feature List', expanded=False):
                st.table(st.session_state.df.columns.to_list())

        for msg in st.session_state.cb_messages:
            st.chat_message(msg["role"]).write(msg["content"])

        if prompt := st.chat_input(placeholder="Ask a question about the DataTable..."):
            st.session_state.cb_messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            llm = model = OpenAI(temperature=0.7, openai_api_key=st.session_state['openai_api_key'])

            pandas_df_agent = create_pandas_dataframe_agent(
                llm,
                st.session_state.df,
                verbose=True,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                handle_parsing_errors=True,
                allow_dangerous_code=True
            )

            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                response = pandas_df_agent.run(st.session_state.cb_messages, callbacks=[st_cb])

                st.session_state.cb_messages.append({"role": "assistant", "content": response})
                st.write(response)
else:
    st.chat_input("Load Data First!", disabled=True)
