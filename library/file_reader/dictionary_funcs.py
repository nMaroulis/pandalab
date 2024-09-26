import json
import os

# def create_default_dictionary(dict_name='default'):
#     global default_feature_labels
#     json_file_path = './library/file_reader/dictionaries/' + dict_name+'_dictionary.json'
#     with open(json_file_path, "w+") as fp:
#         json.dump(default_feature_labels, fp)
#     return 1


def create_new_dictionary(dict_name='default'):
    json_file_path = './library/file_reader/dictionaries/' + dict_name+'_dictionary.json'
    with open(json_file_path, "w+") as fp:
        json.dump({"old label": "new label"}, fp)
    return 1


def get_dictionary(dict_name='default'):
    json_file_path = './library/file_reader/dictionaries/' + dict_name + '_dictionary.json'
    with open(json_file_path, "r") as fp:
        dictionary_dict = json.load(fp)
    return dictionary_dict


def update_dictionary(dict_name='default', dict_df=None):

    dict_data = dict(dict_df[["Old Column", "New Column"]].values) # from Dictionary Dataframe to python Dict
    json_file_path = './library/file_reader/dictionaries/' + dict_name + '_dictionary.json'
    with open(json_file_path, "w+") as fp:
        json.dump(dict_data, fp)
    return 1


def get_dictionaries():
    dir_path = './library/file_reader/dictionaries/'
    res = []
    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            res.append(path[0:-16])
    return res
