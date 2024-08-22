#!/scratch/balta0/2263373r/conda_envs/giflow/bin/python
import torch
import numpy as np
import os
import pickle as pkl
import json
import pandas as pd
import sys

from giflow.results import BoxFlowResults
from giflow.flowmodel import FlowModel
from giflow.plot import plot_js_hist
from giflow.read_files import read_files
from giflow.latent import FlowLatent

#n = int(sys.argv[1])
#survey_coordinates_to_include = []
survey_coordinates_to_include = ['x', 'y', 'noise_scale']
model_info_to_include= []
mix_survey_order = False
#bilby_location = '/data/www.astro/2263373r/giflow/4_paper/bilby/'
bilby_location = None
flow_location = '/data/www.astro/2263373r/giflow/box/combined/run_2024-08-21 10:27:06.950900/'

#directories = []
#for roots, dirs, files in os.walk(flow_location):
#    for dir in dirs:
#        directories.append(dirs)
#directories = directories[0]
#print(directories)
#
#flow_location = os.path.join(flow_location, directories[n])
# -------------------- Reading the flow --------------------------
device = torch.device('cuda')
flow=FlowModel()
flow.load(flow_location)
flow.flowmodel.to(device)
flow.save_location = flow_location
data_location = flow.data_location


# -------------------- Validation data --------------
with open(os.path.join(flow.data_location, "validationset_0.pkl"), 'rb') as file:
#with open("/data/wiay/2263373r/giflow/box/qinetiq_dummy_set.pkl", 'rb') as file:
    dt_val = pkl.load(file)
keys = dt_val.parameter_labels
labels = [r'$c_x$', r'$c_y$', r'$c_z$', r'$l_x$', r'$l_y$', r'$l_z$', r'$\alpha$']
priors = dt_val.priors

prior_bounds = []
for k in keys:
    p = priors.distributions[k]
    prior_bounds.append([p[1], p[2]])


valsize = 100000 # THIS needs to be edited to give the overall desired data set size
val_data, val_conditional = read_files(data_location=data_location, filenames=['validationset_0.pkl'], datasize=valsize, survey_coordinates_to_include=survey_coordinates_to_include, model_info_to_include=model_info_to_include, mix_survey_order=mix_survey_order)

val_dataset = flow.make_tensor_dataset(val_data, val_conditional, device=device, scale=True)

# -------------------- Test data  ------------------
with open(os.path.join(flow.data_location, "testset_to_present_0.pkl"), 'rb') as file:
#with open("/data/wiay/2263373r/giflow/box/qinetiq_dummy_set.pkl", 'rb') as file:
    dt_test = pkl.load(file)

testsize = 4 # THIS needs to be edited to give the overall desired data set size
test_data, test_conditional = read_files(data_location=data_location, filenames=['testset_to_present_0.pkl'], datasize=testsize, survey_coordinates_to_include=survey_coordinates_to_include, model_info_to_include=model_info_to_include, mix_survey_order=mix_survey_order)


test_dataset = flow.make_tensor_dataset(test_data, test_conditional, device=device, scale=True)

# -------------------- PP data -----------------------
with open(os.path.join(flow.data_location, "ppset_0.pkl"), 'rb') as file:
#with open("/data/wiay/2263373r/giflow/box/qinetiq_dummy_set.pkl", 'rb') as file:
    dt_pp = pkl.load(file)
ppsize = 100 # THIS needs to be edited to give the overall desired data set size
pp_data, pp_conditional = read_files(data_location=data_location, filenames=['ppset_0.pkl'], datasize=ppsize, survey_coordinates_to_include=survey_coordinates_to_include, model_info_to_include=model_info_to_include, mix_survey_order=mix_survey_order)

pp_dataset = flow.make_tensor_dataset(pp_data, pp_conditional, device=device, scale=True)
# ------------------- Sampling ---------------------------------
results = []
for i in range(testsize):
    samples, log_probabilities = flow.sample_and_logprob(test_dataset.tensors[1][i], num=2000)
    result = BoxFlowResults(samples=samples, conditional=[test_conditional[j][i] for j in range(len(test_conditional))], log_probabilities=log_probabilities, true_parameters=np.array([test_data[0][i]]), parameter_labels=keys, survey_coordinates=dt_test.surveys[0].survey_coordinates)
    result.directory = os.path.join(flow_location, f"testcase_{i}/")
    result.samples_to_csv()
    dt_test.surveys[i].plot_contours(filename=os.path.join(result.directory, "survey.png"), include_noise=True)
    results.append(result)

# ----------------- Consistency tests --------------------------
# P-P TEST
flow.pp_test(validation_dataset=pp_dataset,
    parameter_labels = labels,
    #parameter_labels=[r'$q_1$', r'$q_2$', r'$q_3$', r'$q_4$', r'$q_5$', r'$q_6$', r'$q_7$', r'$q_8$', r'$q_9$', r'$q_10$']
)

# CORNER PLOTS                              
for i, result in enumerate(results):
    result.corner_plot(filename="corner_plot.png")
    result.corner_plot(filename="corner_plot_with_prior_bounds.png", prior_bounds=prior_bounds)
    print(f"Made {i+1}/{testsize} corner plots.")

# SURVEY CONSISTENCY
for i, result in enumerate(results):
    result.plot_compare_surveys(model_framework=dt_test.model_framework, filename="compare_survey.png", include_examples=True)
    print(f"Made {i+1}/{testsize} survey comparison plots.")
#
## VOXELISED MODEL COMPARISON
#for i, result in enumerate(results):
#    result.plot_compare_voxel_slices(filename=f"compare_voxel_slices.png",
#        plot_truth=True,
#        normalisation=[-1500.0, 500.0], 
#        model_framework=dt_test.model_framework,
#        slice_coords=[1, 4, 8])
#    print(f"Made {i+1}/{testsize} voxel slice comparison plots.")
#
## 3D PLOT COMPARISON PLOT
#for i, result in enumerate(results):
#    result.plot_3D_statistics(dt_test.model_framework)
#    result.plot_3D_samples(dt_test.model_framework, mode='maxlikelihood', num_to_plot=100, filename='maxlikelihood_animation.gif')
#
#
## ------------------------ Comparison with Bilby --------------------------------
#if bilby_location is not None:
#    # JS-DIVERGENCE WITH PRIOR
#    js_100_cases = []
#    for i in range(100):
#        samples, log_probabilities = flow.sample_and_logprob(pp_dataset.tensors[1][i], num=2000)
#        result_js = BoxFlowResults(samples=samples, conditional=[pp_conditional[j][i] for j in range(len(pp_conditional))], log_probabilities=log_probabilities, true_parameters=np.array([pp_data[0][i]]), parameter_labels=keys, survey_coordinates=dt_pp.surveys[0].survey_coordinates)
#        js_100_cases.append(priors.get_js_divergence(samples)[0])
#    js_100_cases = np.vstack(js_100_cases).T
#    data = dict.fromkeys(keys)
#    for i, key in enumerate(keys):
#        data[key] = js_100_cases[i, :]
#    df = pd.DataFrame(data=data)
#    df.to_csv(os.path.join(flow_location, 'js_divergence_with_prior.csv'))
#
#    js_100_cases_bilby = []
#    for i in range(100):
#        samples, log_probabilities = flow.sample_and_logprob(pp_dataset.tensors[1][i], num=2000)
#        result_js = BoxFlowResults(samples=samples, conditional=[pp_conditional[j][i] for j in range(len(pp_conditional))], log_probabilities=log_probabilities, true_parameters=np.array([pp_data[0][i]]), parameter_labels=keys, survey_coordinates=dt_pp.surveys[0].survey_coordinates)
#        result_js.directory = os.path.join(flow_location, '100_testcases/')
#        with open(os.path.join(bilby_location, "100_testcases", f"testcase_{i}", "inversion_result.json"), 'r') as file:
#            bilby_results = json.load(file)
#            bilby_posterior_dict = bilby_results['posterior']['content']
#            bilby_samples = []
#            for key in keys:
#                bilby_samples.append(bilby_posterior_dict[key])
#            bilby_samples = np.array(bilby_samples).T
#        result_js.overlaid_corner(bilby_samples, ['Nested Sampling', 'Normalising Flow'], parameter_labels=labels, filename=f"overlaid_corner_plot_{i}.png")
#        js_100_cases_bilby.append(result_js.get_js_divergence(bilby_samples))
#    js_100_cases_bilby = np.vstack(js_100_cases_bilby).T
#    _, _, median = plot_js_hist(js_100_cases_bilby, keys=keys, filename=os.path.join(flow_location, 'js_divergence_with_bilby_hist.png'))
#    data = dict.fromkeys(keys)
#    for i, key in enumerate(keys):
#        data[key] = js_100_cases_bilby[i, :]
#    df = pd.DataFrame(data=data)
#    df.to_csv(os.path.join(flow_location, 'js_divergences_with_bilby.csv'))
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
