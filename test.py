import torch
import numpy as np
import os
import pickle as pkl
import json
import pandas as pd
import json

from giflow.results import FaultFlowResults
from giflow.flowmodel import FlowModel
from giflow.datareader import DataReader
from giflow.prior import Prior
flow_location = '/data/www.astro/2263373r/fault_python_version/real_inversion_12_param/run_2025-08-14 09:45:20.364071/'


#Reading the flow
device = torch.device('cuda')
flow = FlowModel()
flow.load(flow_location)
flow.flowmodel.to(device)
flow.save_location = flow_location
data = flow.data_location

# ------------------- DATA -------------------------
#model_info_to_include = ['cx', 'cy', 'cz', 'l', 'alpha', 'DL_ratio']
#labels = [r'$c_x$', r'$c_y$', r'$c_z$', r'$l$', r'$\alpha$', r'$\gamma$']
#survey_info_to_include = ['survey_width_ratio', 'noise_scale', 'density']
#noise_distribution = Prior(distributions={'noise_scale': ['LogUniform', 1e-5, 2.0]})
model_info_to_include = ['cx', 'cy', 'l', 'alpha', 'cz', 'DL_ratio', 'dip', 'density', 'Displacement_order', 'Blend_order', 'sym_factor', 'Extent_ratio']
survey_info_to_include = ['survey_width_ratio', 'noise_scale']
labels = [r'$c_x$', r'$c_y$', r'$l$', r'$\alpha$', r'$c_z$', r'$\gamma$', r'$\beta$', r'$\rho$', r'$o_d$', r'$o_b$', r'$a$', r'$E$']

noise_distribution = Prior(distributions={'noise_scale': ['LogUniform', 1e-5, 1.0]})
# Reading in files
ppsize = 100
dr_pp = DataReader(filenames="ppset_0.pkl", data_location=data, model_info_to_include=model_info_to_include, survey_info_to_include=survey_info_to_include, datasize=ppsize)
pp_data, pp_conditional = dr_pp.read_files(noise_distribution=noise_distribution, noise_seed=123)


if noise_distribution.distributions['noise_scale'][0] == 'LogUniform':
    pp_conditional[2] = np.log(pp_conditional[2])

pp_dataset = flow.make_tensor_dataset(pp_data, pp_conditional, device=device, scale=True)

testsize = 10
dr_test = DataReader(filenames="testset_0.pkl", data_location=data, model_info_to_include=model_info_to_include, survey_info_to_include=survey_info_to_include, datasize=testsize)
test_data, test_conditional = dr_test.read_files(noise_distribution=noise_distribution, noise_seed=123)
#noise_scale = test_conditional[2]

if noise_distribution.distributions['noise_scale'][0] == 'LogUniform':
    test_conditional[2] = np.log(test_conditional[2])



test_dataset = flow.make_tensor_dataset(test_data, test_conditional, device=device, scale=True)

with open(os.path.join(data, 'testset_0.pkl'), 'rb') as file:
    dt_test = pkl.load(file)
for s in dt_test.surveys:
    s.make_survey()

with open(os.path.join(data, 'model_framework.json'), 'r') as file:
    model_framework = json.load(file)

with open(os.path.join(data, 'survey_framework.json'), 'r') as file:
    survey_framework = json.load(file)

with open(os.path.join(data, 'prior.pkl'), 'rb') as file:
    priors = pkl.load(file)

prior_bounds = [priors.distributions[k][1:] for k in model_info_to_include]

# --------------- TESTING -------------------
# P-P plot
flow.pp_test(validation_dataset=pp_dataset,
            parameter_labels=labels,
            num_params=12)


# Generating results for some test data
results = []
for i in range(testsize):
    samples, log_probabilities = flow.sample_and_logprob(test_dataset.tensors[1][i], # the conditional
                                                         num=10000) # number of samples we want to draw
    samples = np.array([s[:,0] for s in samples]).T
    result = FaultFlowResults(samples=samples,
                            conditional=[test_conditional[j][i] for j in range(len(test_conditional))],
                            log_probabilities=log_probabilities,
                            true_parameters=np.array([td[i,0] for td in test_data]).T,
                            parameter_labels=labels,
                            survey_coordinates=np.reshape(dt_test.surveys[i].survey_coordinates, (dt_test.surveys[i].shape[0], dt_test.surveys[i].shape[1], 3))
                           )
    results.append(result)
    result.directory = os.path.join(flow_location, f"testcase_{i}/")
    # plotting the surveys we are inverting
    dt_test.surveys[i].plot_contours(filename=os.path.join(result.directory, "survey_contours.png"))
    #dt_test.surveys[i].plot_pixels(filename=os.path.join(result.directory, "survey.png"))

# CORNER PLOTS                              
#for i, result in enumerate(results):
#    result.corner_plot(filename="corner_plot.png")
#    result.corner_plot(filename="corner_plot_prior.png", prior_bounds=prior_bounds)

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


## Comparison with bilby
#
#bilby_loc = '/data/www.astro/2263373r/fault_python_version/bilby/5_parameter_test/5_parameter_test_result.json'
#
#with open(bilby_loc, 'rb') as file:
#    bilby_results = json.load(file)
#    bilby_results = bilby_results['posterior']['content']
#bilby_samples = []
#for k in model_info_to_include:
#    bilby_samples.append(np.expand_dims(bilby_results[k], axis=1))
#bilby_samples = np.hstack(bilby_samples)
#print(np.shape(bilby_samples))
#result.overlaid_corner(bilby_samples, ['Nested Sampling', 'Normalising Flow'], parameter_labels=labels, filename='bilby_compare.png')
#

for i, result in enumerate(results):
    result.plot_compare_surveys_samples(model_framework=model_framework, model_info_to_include=model_info_to_include, filename='survey_compare_samples.png', priors=None)
    result.plot_compare_surveys_v2(model_framework=model_framework, filename='survey_compare.png', priors=None, units='mGal', window=('tukey', 0.1))
