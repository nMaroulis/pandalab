import streamlit as st
from pandas import eval as pd_eval


def feature_generation_form():
    with st.form("data_generation"):

        new_feature_name = st.text_input('(1) New Feature Name', placeholder="Currently the following characters are supported a-z, _, 0-9", help="Do not use spaces or special characters")


        st.write("The ***formula*** is an equation from which the new Feature's value will take its value. The ***Syntax*** needs to follow specific rules and it is explained below:")
        with st.expander('Syntax Guide', expanded=True):
            st.write("The Formula Syntax must be **strictly followed**, in any other case there will be an error.")

            st.markdown("<br>", unsafe_allow_html=True)
            st.write("**Variable Assignment:**")
            st.write("> The User must define from the selecteboxes above, which features will correspond to each variable (up to 5). If a variable isn't assigned a Feature and remains empty, it will not be used and should not be added into the formula. E.g. If a new Feature which is the aveage of two other Features needs to be created, assign the first Feature to var1 and the second to var2, leave the others empty.")

            st.markdown("<br>", unsafe_allow_html=True)
            st.write("**Supported Math Operations:**")
            st.code("""+, -, /, *, ^, (, ), <, <=, >, >=, % (mod), abs, sqrt, exp, sin, cos, log, expm1, log1p, sinh, cosh, tanh, arcsin, arccos, arctan, arccosh, arcsinh, arctanh, arctan2, log10.""")

            st.markdown("<br>", unsafe_allow_html=True)
            st.write("**Examples:**")
            st.write("> New feature is the sum of Variable 1 and Variable 2  →  :orange[**var1 + var2**]")
            st.write("> New feature is the mean/average of Variable 1, 2 and 3  →  :orange[**(var1 + var2 + var3) / 3**]")
            st.write("> New feature is the True/False if a condition is met/not met  →  :orange[**100 >= var1 > 50**]")
            st.write("> New feature is a combination of other features  → :orange[**abs(var1 - var2) + 0.243 + var3/var4*sqrt(var1)**]")


        st.write("(2) Choose Variables (Features):")
        new_f_feats = list(st.session_state.df._get_numeric_data().columns)
        new_f_feats.insert(0, "<empty>")
        new_f_cols = st.columns(5)
        with new_f_cols[0]:
            var1 = st.selectbox('Variable ***var1***', options=new_f_feats)
        with new_f_cols[1]:
            var2 = st.selectbox('Variable ***var2***', options=new_f_feats)
        with new_f_cols[2]:
            var3 = st.selectbox('Variable ***var3***', options=new_f_feats)
        with new_f_cols[3]:
            var4 = st.selectbox('Variable ***var4***', options=new_f_feats)
        with new_f_cols[4]:
            var5 = st.selectbox('Variable ***var5***', options=new_f_feats)

        new_feature_formula = st.text_area("(3) Formula", placeholder='Define Formula here...')
        submitted_dg = st.form_submit_button("Generate Features")
        if submitted_dg:
            if len(new_feature_formula) <= 0 or len(new_feature_formula) <= 0:
                st.warning('No Data')
            else:
                if var1 != "<empty>":
                    new_feature_formula = new_feature_formula.replace("var1", "st.session_state.df['" + var1 + "']")
                if var2 != "<empty>":
                    new_feature_formula = new_feature_formula.replace("var2", "st.session_state.df['" + var2 + "']")
                if var3 != "<empty>":
                    new_feature_formula = new_feature_formula.replace("var3", "st.session_state.df['" + var3 + "']")
                if var4 != "<empty>":
                    new_feature_formula = new_feature_formula.replace("var4", "st.session_state.df['" + var4 + "']")
                if var5 != "<empty>":
                    new_feature_formula = new_feature_formula.replace("var5", "st.session_state.df['" + var5 + "']")
                new_feature_formula = new_feature_name + " = " + new_feature_formula

                print("Feature Generation Formula", new_feature_formula)
                with st.spinner('Generating new Feature'):

                    try:
                        pd_eval(expr=new_feature_formula, target=st.session_state.df, inplace=True)
                        st.session_state['generated features'] += 1
                        st.success('Feature ***' + new_feature_name + '*** was generated Successfully')

                        new_df_col = [new_feature_name]

                        if var1 != "<empty>": new_df_col.append(var1)
                        if var2 != "<empty>": new_df_col.append(var2)
                        if var3 != "<empty>": new_df_col.append(var3)
                        if var4 != "<empty>": new_df_col.append(var4)
                        if var5 != "<empty>": new_df_col.append(var5)
                        st.cache_data.clear()  # CLEAR CACHE
                        st.toast("✅ Feature Generation Successful")
                    except Exception as e:
                        print(e)
                        st.error('Feature Generation Failed! Check Formula Parameters')
                        return

                    st.write('Overview')
                    print(new_df_col)
                    st.dataframe(st.session_state.df[new_df_col].head(), use_container_width=True)
