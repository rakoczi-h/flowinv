#!/scratch/wiay/2263373r/masters/conda_envs/venv/bin/python

import torch
import numpy as np
import os
import pickle as pkl
import json

from giflow.results import BoxFlowResults
from giflow.flowmodel import FlowModel
from giflow.plot import plot_js_hist
from giflow.read_files import read_files

survey_coordinates_to_include = ['noise_scale']
num_test_cases = 10
bilby_location = '/data/www.astro/2263373r/giflow/bilby/box/normalised/'
#bilby_location = None
flow_location = '/data/www.astro/2263373r/giflow/box/parameterised/normalised/run_2024-06-05 14:23:51.682263/'


# -------------------- Reading the flow --------------------------
device = torch.device('cuda')
flow=FlowModel()
flow.load(flow_location)
flow.flowmodel.to(device)
data_location = flow.data_location

# -------------------- Validation data --------------
valsize = 100000 # THIS needs to be edited to give the overall desired data set size
num_files = 1 #number of files that needs to be read
val_data, val_conditional = read_files(data_location=data_location, filename='validationset', datasize=valsize, num_files=num_files, survey_coordinates_to_include=survey_coordinates_to_include)

val_dataset = flow.make_tensor_dataset(val_data, val_conditional, device=device, scale=True)

# -------------------- Test data  ------------------
testsize = 10 # THIS needs to be edited to give the overall desired data set size
num_files = 1 #number of files that needs to be read
test_data, test_conditional = read_files(data_location=data_location, filename='testset', datasize=testsize, num_files=num_files, survey_coordinates_to_include=survey_coordinates_to_include)

test_dataset = flow.make_tensor_dataset(test_data, test_conditional, device=device, scale=True)

# ------------------- Sampling ---------------------------------
results = []
for i in range(num_test_cases):
    samples, log_probabilities = flow.sample_and_logprob(test_dataset.tensors[1][i], num=2000)
    result = BoxFlowResults(samples=samples, conditional=test_conditional[i,:], log_probabilities=log_probabilities, true_parameters=test_data[i,:], parameter_labels=keys, survey_coordinates=dt_test.surveys[i].survey_coordinates)
    result.directory = os.path.join(flow_location, f"testcase_{i}/")
    dt_test.surveys[i].plot_contours(filename=os.path.join(result.directory, "survey.png"), include_noise=True)
    results.append(result)
#
# ----------------- Consistency tests --------------------------
## P-P TEST
flow.pp_test(validation_dataset=val_dataset)
#
# CORNER PLOTS
for i, result in enumerate(results):
    result.corner_plot(filename="corner_plot.png")
    print(f"Made {i+1}/{num_test_cases} corner plots.")

# SURVEY CONSISTENCY
for i, result in enumerate(results):
    result.plot_compare_surveys(model_framework=dt_test.model_framework, filename="compare_survey.png", include_examples=True)
    print(f"Made {i+1}/{num_test_cases} survey comparison plots.")

# VOXELISED MODEL COMPARISON
#for i, result in enumerate(results):
#    result.plot_compare_voxel_slices(filename=f"compare_voxel_slices.png")
#    print(f"Made {i+1}/{num_test_cases} voxel slice comparison plots.")

# ------------------------ Comparison with Bilby --------------------------------
if bilby_location is not None:
    # JS-DIVERGENCE WITH BILBY (done with validation data)
#    js_100_cases = []
#    for i in range(100):
#        samples, log_probabilities = flow.sample_and_logprob(val_dataset.tensors[1][i], num=2000)
#        result = BoxFlowResults(samples=samples, conditional=val_conditional[i,:], log_probabilities=log_probabilities, true_parameters=val_data[i,:], parameter_labels=keys)
#        with open(os.path.join(bilby_location, f"100_cases/testcase_{i}/box_parameterised_result.json"), 'r') as file:
#            bilby_results = json.load(file)
#            bilby_posterior_dict = bilby_results['posterior']['content']
#            bilby_samples = []
#            for key in keys:
#                bilby_samples.append(bilby_posterior_dict[key])
#            bilby_samples = np.array(bilby_samples).T
#        js_100_cases.append(result.get_js_divergence(bilby_samples))
#    js_100_cases = np.vstack(js_100_cases).T
#    _, _, median = plot_js_hist(js_100_cases, keys=keys, filename=os.path.join(flow_location, 'js_divergence_hist.png'))
#    print(f"The median JS-divergence value is {median}")
#
    # CORNER PLOT WITH BILBY (done with test data)
    for i, result in enumerate(results):
        with open(os.path.join(bilby_location, f"testcase_{i}/inversion_result.json"), 'r') as file:
            bilby_results = json.load(file)
            bilby_posterior_dict = bilby_results['posterior']['content']
            bilby_samples = []
            for key in keys:
                bilby_samples.append(bilby_posterior_dict[key])
            bilby_samples = np.array(bilby_samples).T
        result.overlaid_corner(bilby_samples, ['Flow', 'Dynesty'], filename="overlaid_corner_plot.png")
