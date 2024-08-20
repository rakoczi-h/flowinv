import pickle as pkl
import numpy as np
import os


def read_files(data_location, filenames, datasize, survey_coordinates_to_include=[], model_info_to_include=[], mix_survey_order=False):
    train_data, train_conditional = ([],[])
    if isinstance(filenames, str):
        filenames = [filenames]
    for f in filenames:
        with open(os.path.join(data_location, f), 'rb') as file:
            dt = pkl.load(file)
            td, tc = dt.make_data_for_network(survey_coordinates_to_include=survey_coordinates_to_include, model_info_to_include=model_info_to_include, mix_survey_order=mix_survey_order)
            train_data.append(td)
            train_conditional.append(tc)
        print(f"{f} read")

    tc_list = []
    for i in range(len(train_conditional[0])):
        tc = []
        for j in range(len(train_conditional)):
            tc.append(train_conditional[j][i])
        tc = np.vstack(tc)[:datasize, :]
        tc_list.append(tc)
    tc = []
    train_conditional = tc_list

    td_list = []
    for i in range(len(train_data[0])):
        td = []
        for j in range(len(train_data)):
            td.append(train_data[j][i])
        td = np.vstack(td)[:datasize, :]
        td_list.append(td)
    td = []
    train_data = td_list
    return train_data, train_conditional

