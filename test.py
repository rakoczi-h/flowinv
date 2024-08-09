#!/scratch/wiay/2263373r/masters/conda_envs/venv/bin/python

import torch
import numpy as np
import os
import pickle as pkl
import json
import pandas as pd

from giflow.results import BoxFlowResults
from giflow.flowmodel import FlowModel
from giflow.plot import plot_js_hist
from giflow.read_files import read_files
from giflow.latent import FlowLatent

#survey_coordinates_to_include = []
survey_coordinates_to_include = ['x', 'y', 'noise_scale']
model_info_to_include= []
mix_survey_order = False
#bilby_location = '/data/www.astro/2263373r/giflow/4_paper/bilby/'
bilby_location = None
flow_location = '/data/www.astro/2263373r/giflow/4_paper/voxelised_noisy/run_2024-07-30 20:56:45.690508'

# -------------------- Reading the flow --------------------------
device = torch.device('cuda')
flow=FlowModel()
flow.load(flow_location)
flow.flowmodel.to(device)
flow.save_location = flow_location
data_location = flow.data_location


# -------------------- Validation data --------------
valsize = 100000 # THIS needs to be edited to give the overall desired data set size
num_files = 1 #number of files that needs to be read
val_data, val_conditional = read_files(data_location=data_location, filename='validationset', datasize=valsize, num_files=num_files, survey_coordinates_to_include=survey_coordinates_to_include, model_info_to_include=model_info_to_include, mix_survey_order=mix_survey_order)

val_dataset = flow.make_tensor_dataset(val_data, val_conditional, device=device, scale=True)

# -------------------- Test data  ------------------
with open(os.path.join(flow.data_location, "testset_to_present_0.pkl"), 'rb') as file:
#with open("/data/wiay/2263373r/giflow/box/qinetiq_dummy_set.pkl", 'rb') as file:
    dt_test = pkl.load(file)
keys = dt_test.parameter_labels
labels = [r'$c_x$', r'$c_y$', r'$c_z$', r'$l_x$', r'$l_y$', r'$l_z$', r'$\alpha$']
priors = dt_test.priors

prior_bounds = []
for k in keys:
    p = priors.distributions[k]
    prior_bounds.append([p[1], p[2]])

testsize = 4 # THIS needs to be edited to give the overall desired data set size
num_files = 1 #number of files that needs to be read
test_data, test_conditional = read_files(data_location=data_location, filename='testset_to_present', datasize=testsize, num_files=num_files, survey_coordinates_to_include=survey_coordinates_to_include, model_info_to_include=model_info_to_include, mix_survey_order=mix_survey_order)


test_dataset = flow.make_tensor_dataset(test_data, test_conditional, device=device, scale=True)
print(test_dataset.tensors[1].shape)
# -------------------- PP data -----------------------

with open(os.path.join(flow.data_location, "ppset_0.pkl"), 'rb') as file:
    dt_pp = pkl.load(file)

ppsize = 100 # THIS needs to be edited to give the overall desired data set size
num_files = 1 #number of files that needs to be read
pp_data, pp_conditional = read_files(data_location=data_location, filename='ppset', datasize=ppsize, num_files=num_files, survey_coordinates_to_include=survey_coordinates_to_include, model_info_to_include=model_info_to_include, mix_survey_order=mix_survey_order)
if np.isinf(pp_conditional[0]).any():
    print('found inf')

pp_dataset = flow.make_tensor_dataset(pp_data, pp_conditional, device=device, scale=True)
# ------------------- Sampling ---------------------------------
results = []
for i in range(testsize):
    samples, log_probabilities = flow.sample_and_logprob(test_dataset.tensors[1][i], num=2000)
    result = BoxFlowResults(samples=samples, conditional=[test_conditional[j][i] for j in range(len(test_conditional))], log_probabilities=log_probabilities, true_parameters=np.array([test_data[0][i]]), parameter_labels=keys, survey_coordinates=dt_test.surveys[0].survey_coordinates)
    result.directory = os.path.join(flow_location, f"testcase_to_present_{i}/")
    dt_test.surveys[i].plot_contours(filename=os.path.join(result.directory, "survey.png"), include_noise=True)
    results.append(result)

# ----------------- Consistency tests --------------------------
## P-P TEST
#flow.pp_test(validation_dataset=pp_dataset, parameter_labels=[r'$q_1$', r'$q_2$', r'$q_3$', r'$q_4$', r'$q_5$', r'$q_6$', r'$q_7$', r'$q_8$', r'$q_9$', r'$q_10$'])

## CORNER PLOTS                              
#for i, result in enumerate(results):
#    result.corner_plot(filename="corner_plot.png")
#   #result.corner_plot(filename="corner_plot_with_prior_bounds.png", prior_bounds=prior_bounds)
#    print(f"Made {i+1}/{num_test_cases} corner plots.")

## SURVEY CONSISTENCY
#for i, result in enumerate(results):
#    result.plot_compare_surveys(model_framework=dt_test.model_framework, filename="compare_survey.png", include_examples=False)
#    print(f"Made {i+1}/{testsize} survey comparison plots.")

## VOXELISED MODEL COMPARISON
#for i, result in enumerate(results):
#    result.plot_compare_voxel_slices(filename=f"compare_voxel_slices.png", plot_truth=True, normalisation=[-2500.0, 500.0], model_framework=dt_test.model_framework, slice_coords=[1, 4, 8])
#    print(f"Made {i+1}/{testsize} voxel slice comparison plots.")

# 3D PLOT COMPARISON PLOT
for i, result in enumerate(results):
    #result.plot_3D_statistics(dt_test.model_framework)
    result.plot_3D_samples(dt_test.model_framework, mode='maxlikelihood', num_to_plot=100, filename='cumulative_mean_animation.gif')

#
## ------------------------ Comparison with Bilby --------------------------------
#if bilby_location is not None:
##    # JS-DIVERGENCE WITH BILBY
##    js_100_cases = []
##    snrs = []
##    for i in range(100):
##        snrs.append(dt_pp.surveys[i].snr())
##        samples, log_probabilities = flow.sample_and_logprob(pp_dataset.tensors[1][i], num=2000)
##        result_js = BoxFlowResults(samples=samples, conditional=[pp_conditional[j][i] for j in range(len(pp_conditional))], log_probabilities=log_probabilities, true_parameters=np.array([pp_data[0][i]]), parameter_labels=keys, survey_coordinates=dt_pp.surveys[0].survey_coordinates)
##        result_js.directory = os.path.join(flow_location, f"deep_testcases/")
##        js_100_cases.append(priors.get_js_divergence(samples))
##js_100_cases = np.vstack(js_100_cases).T
##data = dict.fromkeys(keys)
##for i, key in enumerate(keys):
##    data[key] = js_100_cases[i, :]
##df = pd.DataFrame(data=data)
##df.to_csv(os.path.join(result_js.directory, 'js_divergence_with_prior.csv'))
##
#
##        with open(os.path.join(bilby_location, "100_testcases_deep", f"testcase_{i}", "inversion_result.json"), 'r') as file:
##            bilby_results = json.load(file)
##            bilby_posterior_dict = bilby_results['posterior']['content']
##            bilby_samples = []
##            for key in keys:
##                bilby_samples.append(bilby_posterior_dict[key])
##            bilby_samples = np.array(bilby_samples).T
##        result_js.directory = os.path.join(flow_location, f"deep_testcases/")
##        result_js.overlaid_corner(bilby_samples, ['Nested Sampling', 'Normalising Flow'], parameter_labels=labels, filename=f"overlaid_corner_plot_{i}.png")
##        js_100_cases.append(result_js.get_js_divergence(bilby_samples))
##    js_100_cases = np.vstack(js_100_cases).T
##    _, _, median = plot_js_hist(js_100_cases, keys=keys, filename=os.path.join(flow_location, 'js_divergence_hist_deep.png'))
##    data = dict.fromkeys(keys)
##    for i, key in enumerate(keys):
##        data[key] = js_100_cases[i, :]
##    df = pd.DataFrame(data=data)
##    df.to_csv(os.path.join(flow_location, 'js_divergences_deep.csv'))
#
#    # CORNER PLOT WITH BILBY (done with test data)
#    for i, result in enumerate(results):
#        if i == 0 or i == 1 or i == 2:
#            continue
#        with open(os.path.join(bilby_location, "testcases_to_present_v2", f"testcase_{i}", "inversion_result.json"), 'r') as file:
#            bilby_results = json.load(file)
#            bilby_posterior_dict = bilby_results['posterior']['content']
#            bilby_samples = []
#            for key in keys:
#                bilby_samples.append(bilby_posterior_dict[key])
#            bilby_samples = np.array(bilby_samples).T[:,3:5]
#        result.overlaid_corner(bilby_samples, ['NS', 'NF'], parameter_labels=[r'$l_x$', r'$l_y$'], filename="overlaid_corner_plot.png")
#        result.overlaid_corner(bilby_samples, ['NS', 'NF'], parameter_labels=[r'$l_x$', r'$l_y$'], filename="overlaid_corner_plot_with_rior_bounds.png", prior_bounds=[prior_bounds[3], prior_bounds[4]])
#        js = result.get_js_divergence(bilby_samples)
#        print(js)
