#!/scratch/wiay/2263373r/masters/conda_envs/venv/bin/python

import torch
import numpy as np
import os
import pickle as pkl
import json
import h5py
from sklearn.preprocessing import MinMaxScaler

from giflow.results import BoxFlowResults
from giflow.flowmodel import FlowModel
from giflow.plot import plot_js_hist
from giflow.plot import compare_method_surveys

#survey_coordinates_to_include_list = [['x', 'y']]
survey_coordinates_to_include_list = [['x', 'y', 'noise_scale']]
test_case = 2
device = torch.device('cuda')

#flow_locations = ['/data/www.astro/2263373r/giflow/box/voxelised/noisy_grid/run_2024-03-29 19:38:12.842048/']
#flow_locations = ['/data/www.astro/2263373r/giflow/4_paper/parameterised/run_2024-07-26 13:50:53.730077/', '/data/www.astro/2263373r/giflow/4_paper/voxelised_noisy/run_2024-07-30 20:56:45.690508/']
#flow_locations = ['/data/www.astro/2263373r/giflow/4_paper/voxelised_noisy/run_2024-07-30 20:56:45.690508/']
flow_locations = ['/data/www.astro/2263373r/giflow/4_paper/combined/run_2024-10-14 10:44:55.499724/']
# -------------------- Reading the flow --------------------------
results = []
model_frameworks = []
survey_frameworks = []
for idx, flow_location in enumerate(flow_locations):
    flow=FlowModel()
    flow.load(flow_location)
    flow.flowmodel.to(device)
    #flow.data_location = '/scratch/balta1/2263373r/4_paper/voxelised_noisy/'
    print('train data read')
    with open(os.path.join(flow.data_location, "testset_to_present_0.pkl"), 'rb') as file:
        dt_test = pkl.load(file)
    if idx == 0:
        keys = dt_test.parameter_labels
    else:
        keys = None
    model_frameworks.append(dt_test.model_framework)
    survey_frameworks.append(dt_test.survey_framework)
    test_data, test_conditional = dt_test.make_data_for_network(survey_coordinates_to_include=survey_coordinates_to_include_list[idx])
    test_dataset = flow.make_tensor_dataset(test_data, test_conditional, device=device, scale=True)

    print('test data read')
    samples, log_probabilities = flow.sample_and_logprob(test_dataset.tensors[1][test_case], num=2000)

    result = BoxFlowResults(samples=samples, conditional=[test_conditional[j][test_case] for j in range(len(test_conditional))], log_probabilities=log_probabilities, true_parameters=test_data[0][test_case,:], parameter_labels=dt_test.parameter_labels, survey_coordinates=dt_test.surveys[test_case].survey_coordinates)
    results.append(result)
#    with open(os.path.join(flow_location, 'result_5.pkl'), 'rb') as file:
#        result = pkl.load(file)
#    results.append(result)

compare_method_surveys([results[0]], [model_frameworks[0]], [survey_frameworks[0]], filename='/data/www.astro/2263373r/giflow/4_paper/survey_comparison_plot_4_paper_combined.png')
