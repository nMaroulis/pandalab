from streamlit import markdown, sidebar, data_editor, form, form_submit_button, success, error, code, text_input, toast
# from settings.settings import dashboard_version
from library.file_reader.dictionary_funcs import get_dictionary, update_dictionary, create_new_dictionary, get_dictionaries
from pandas import DataFrame
from library.overview.navigation import page_header


page_header("Dictionary")
markdown("<h2 style='text-align: center; color: #373737;'>Old Column Label - Data table New Column Label Mapping</h2>", unsafe_allow_html=True)

dict_options = get_dictionaries()
dictionary_choice = sidebar.selectbox('Dictionary Catalog:', options=dict_options)
sidebar.caption('New Dictionary:')

with sidebar.form('new_dict'):
    new_dict_name = text_input('Dictionary Name', placeholder='Insert here..')
    new_dict_button = form_submit_button('Create New')
    if new_dict_button:
        create_new_dictionary(new_dict_name)
        success("Dictionary **" + new_dict_name + "** was created successfully, please refresh the Tab.")
        toast('✅ Dictionary Created successfully')


def get_dict(dictionary_choice='default'):
    feature_map = get_dictionary(dictionary_choice)
    fm_dict = DataFrame.from_dict(feature_map.items())
    fm_dict.columns = ["Old Column", "New Column"]
    fm_dict.sort_values(by='Old Column', inplace=True)
    fm_dict = fm_dict[["Old Column", "New Column"]].reset_index(drop=True)
    return fm_dict


with form(key="dict_form"):

    markdown("""***Feature Dictionary***""")
    code('Click on the bottom of the Table to add new Feature to the Dictionary. Click on the left side of a row and press Delete, in order to Remove an Entry from the Dictionary.', language="None")
    feature_dictionary = data_editor(get_dict(dictionary_choice), num_rows="dynamic", use_container_width=True, height=800,
                                     hide_index=True)

    # SUBMIT
    markdown("<hr style='text-align: left; width:10em; margin: 1em 0em 1em; color: #5a5b5e'></hr>", unsafe_allow_html=True)
    b_sub = form_submit_button("Update Dictionary")
    if b_sub:
        if update_dictionary(dictionary_choice, feature_dictionary):
            success('Custom Dictionary Labels saved successfully, Refresh to see updated Dictionary!')
        else:
            error('Something went wrong while saving the custom Dictionary. Please check your parameters for Duplicates in the New Column name.')
