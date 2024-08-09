import torch
import numpy as np
import os
import pickle as pkl
import pandas as pd

from giflow.results import BoxFlowResults
from giflow.flowmodel import FlowModel

flow_location = 'trained_flow'

#Reading the flow
device = torch.device('cuda')
flow = FlowModel()
flow.load(flow_location)
flow.flowmodel.to(device)
flow.save_location = flow_location
data = flow.data_location

# Validation data
valsize = 500 
with open(os.path.join(data, 'valset.pkl'), 'rb') as file:
    dt_val = pkl.load(file)
    validation_data, validation_conditional = dt_val.make_data_for_network(
        survey_coordinates_to_include = ['noise_scale'],
        model_info_to_include = [],
        add_noise = True,
        mix_survey_order = False,
    )
validation_dataset = flow.make_tensor_dataset(validation_data, validation_conditional, device=device, scale=True)

# Test data
testsize = 2
with open(os.path.join(data, 'testset.pkl'), 'rb') as file:
    dt_test = pkl.load(file)
    test_data, test_conditional = dt_test.make_data_for_network(
        survey_coordinates_to_include = ['noise_scale'],
        model_info_to_include = [],
        add_noise = True,
        mix_survey_order = False,
    )
test_dataset = flow.make_tensor_dataset(test_data, test_conditional, device=device, scale=True)

# --------------- TESTING -------------------
# P-P plot
flow.pp_test(validation_dataset=validation_dataset,
             parameter_labels=dt_val.parameter_labels)

# Generating results for some test data
results = []
for i in range(testsize):
    samples, log_probabilities = flow.sample_and_logprob(test_dataset.tensors[1][i], # the conditional
                                                         num=2000) # number of samples we want to draw
    result = BoxFlowResults(samples=samples,
                            conditional=[test_conditional[j][i] for j in range(len(test_conditional))],
                            log_probabilities=log_probabilities,
                            true_parameters=np.array([test_data[0][i]]),
                            parameter_labels=dt_val.parameter_labels,
                            survey_coordinates=dt_test.surveys[0].survey_coordinates
                           )
    results.append(result)

    result.directory = os.path.join(flow_location, f"testcase_to_present_{i}/")
    # plotting the surveys we are inverting
    dt_test.surveys[i].plot_contours(filename=os.path.join(result.directory, "survey.png"), include_noise=True)

# CORNER PLOTS                              
for i, result in enumerate(results):
    result.corner_plot(filename="corner_plot.png")

# VOXELISED MODEL COMPARISON
for i, result in enumerate(results):
    result.plot_compare_voxel_slices(filename=f"compare_voxel_slices.png",
                                     plot_truth=True,
                                     normalisation=[-2500.0, 500.0],
                                     model_framework=dt_test.model_framework,
                                     slice_coords=[1, 4, 8])
## 3D PLOT COMPARISON PLOT
#for i, result in enumerate(results):
#    #result.plot_3D_statistics(dt_test.model_framework)
#    result.plot_3D_samples(dt_test.model_framework, mode='maxlikelihood', num_to_plot=100, filename='cumulative_mean_animation.gif')
