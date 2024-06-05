import pickle as pkl
import numpy as np
import os


def read_files(data_location, filename, datasize, num_files, survey_coordinates_to_include=[]):
    train_data, train_conditional = ([],[])
    for n in range(num_files):
        print('file read')
        with open(os.path.join(data_location, filename+f"_{n}.pkl"), 'rb') as file:
            dt = pkl.load(file)
            td, tc = dt.make_data_arrays(survey_coordinates_to_include=survey_coordinates_to_include)
            train_data.append(td)
            train_conditional.append(tc)

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

