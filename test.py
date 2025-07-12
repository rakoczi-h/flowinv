import torch
import numpy as np
import os
import pickle as pkl
import pandas as pd

from giflow.results import FlowResults
from giflow.flowmodel import FlowModel
from giflow.datareader import DataReader

flow_location = '/data/www.astro/2263373r/fault_python_version/5_parameter_test/run_2025-07-07 08:08:14.088743/'

#Reading the flow
device = torch.device('cuda')
flow = FlowModel()
flow.load(flow_location)
flow.flowmodel.to(device)
flow.save_location = flow_location
data = flow.data_location

# ------------------- DATA -------------------------
model_info_to_include = ['cx', 'cy', 'l', 'alpha', 'cz']
labels = [r'$c_x$', r'$c_y$', r'$l$', r'$\alpha$', r'$c_z$']
survey_info_to_include = []
# Reading in files
ppsize = 100
dr_pp = DataReader(filenames="ppset_1.pkl", data_location=data, model_info_to_include=model_info_to_include, survey_info_to_include=survey_info_to_include, datasize=ppsize)
pp_data, pp_conditional = dr_pp.read_files()

pp_dataset = flow.make_tensor_dataset(pp_data, pp_conditional, device=device, scale=True)

testsize = 1
dr_test = DataReader(filenames="testset_1.pkl", data_location=data, model_info_to_include=model_info_to_include, survey_info_to_include=survey_info_to_include, datasize=testsize)
test_data, test_conditional = dr_test.read_files()

test_dataset = flow.make_tensor_dataset(test_data, test_conditional, device=device, scale=True)

with open(os.path.join(data, 'trainset_1.pkl'), 'rb') as file:
    dt_test = pkl.load(file)
print(dt_test.sourcemodels[0].parameters)

# --------------- TESTING -------------------
# P-P plot
flow.pp_test(validation_dataset=pp_dataset,
             parameter_labels=labels)


# Generating results for some test data
results = []
for i in range(testsize):
    samples, log_probabilities = flow.sample_and_logprob(test_dataset.tensors[1][i], # the conditional
                                                         num=2000) # number of samples we want to draw
    samples = np.array([s[:,0] for s in samples]).T
    print(np.shape(samples))
    result = FlowResults(samples=samples,
                            conditional=[test_conditional[j][i] for j in range(len(test_conditional))],
                            log_probabilities=log_probabilities,
                            true_parameters=np.array([td[i,0] for td in test_data]).T,
                            parameter_labels=labels,
                            survey_coordinates=dt_test.surveys[0].survey_coordinates
                           )
    results.append(result)

    result.directory = os.path.join(flow_location, f"testcase_bilby_{i}/")
    # plotting the surveys we are inverting
    dt_test.surveys[i].plot_contours(filename=os.path.join(result.directory, "survey.png"), include_noise=True)
    dt_test.surveys[i].plot_pixels(filename=os.path.join(result.directory, "survey.png"), include_noise=True)

# CORNER PLOTS                              
for i, result in enumerate(results):
    result.corner_plot(filename="corner_plot.png")

# VOXELISED MODEL COMPARISON
#dt_test.model_framework['ranges'] = [[-0.75, 0.75], [-0.75, 0.75], [-1.5, 0.0]]
#dt_test.model_framework['grid_shape'] = [10, 10, 10]
#
#for i, result in enumerate(results):
#    result.plot_compare_voxel_slices(filename=f"compare_voxel_slices.png",
#                                     plot_truth=True,
#                                     normalisation=[-2500.0, 500.0],
#                                     model_framework=dt_test.model_framework,
#                                     slice_coords=[1, 4, 8])
