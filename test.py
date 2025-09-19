import torch
import numpy as np
import os
import pickle as pkl
import json
import pandas as pd
import json
import matplotlib.pyplot as plt

from giflow.results import FaultFlowResults
from giflow.flowmodel import FlowModel
from giflow.datareader import DataReader
from giflow.prior import Prior
flow_location = '/data/www.astro/2263373r/fault_python_version/real_inversion/run_2025-07-31 14:23:46.356947/'


#Reading the flow
device = torch.device('cuda')
flow = FlowModel()
flow.load(flow_location)
flow.flowmodel.to(device)
flow.save_location = flow_location
data = flow.data_location

# ------------------- DATA -------------------------
model_info_to_include = ['cx', 'cy', 'cz', 'l', 'alpha', 'DL_ratio']
labels = [r'$c_x$', r'$c_y$', r'$c_z$', r'$l$', r'$\alpha$', r'$\gamma$']
survey_info_to_include = ['survey_width_ratio', 'noise_scale', 'density']
#model_info_to_include = ['cx', 'cy', 'l', 'alpha', 'cz', 'DL_ratio', 'dip', 'density', 'Displacement_order', 'Blend_order', 'sym_factor', 'Extent_ratio']
#survey_info_to_include = ['survey_width_ratio', 'noise_scale']
#labels = [r'$c_x$', r'$c_y$', r'$l$', r'$\alpha$', r'$c_z$', r'$\gamma$', r'$\beta$', r'$\rho$', r'$o_d$', r'$o_b$', r'$a$', r'$E$']

noise_distribution = Prior(distributions={'noise_scale': ['LogUniform', 1e-5, 1.0]})
# Reading in files
ppsize = 100
dr_pp = DataReader(filenames="ppset_0.pkl", data_location=data, model_info_to_include=model_info_to_include, survey_info_to_include=survey_info_to_include, datasize=ppsize)
pp_data, pp_conditional = dr_pp.read_files(noise_distribution=noise_distribution)


if noise_distribution.distributions['noise_scale'][0] == 'LogUniform':
    pp_conditional[2] = np.log(pp_conditional[2])

pp_dataset = flow.make_tensor_dataset(pp_data, pp_conditional, device=device, scale=True)

testsize = 1
dr_test = DataReader(filenames="testset_2.pkl", data_location=data, model_info_to_include=model_info_to_include, survey_info_to_include=survey_info_to_include, datasize=testsize)
test_data, test_conditional = dr_test.read_files(noise_distribution=Prior(distributions={'noise_scale': [0.025]})
, noise_seed=123)

if noise_distribution.distributions['noise_scale'][0] == 'LogUniform':
    test_conditional[2] = np.log(test_conditional[2])

true_parameters = np.array([td[0,0] for td in test_data]).T

test_dataset = flow.make_tensor_dataset(test_data, test_conditional, device=device, scale=True)

with open(os.path.join(data, 'testset_2.pkl'), 'rb') as file:
    dt_test = pkl.load(file)
s = dt_test.surveys[0]
s.make_survey()
s.noise_scale = 0.025
s.make_noise(seed=123)

with open(os.path.join(data, 'model_framework.json'), 'r') as file:
    model_framework = json.load(file)
model_framework['varied_parameters'] = model_info_to_include
model_framework['default_parameters']['density'] = 800.0
with open(os.path.join(data, 'survey_framework.json'), 'r') as file:
    survey_framework = json.load(file)

with open(os.path.join(data, 'prior.pkl'), 'rb') as file:
    priors = pkl.load(file)

prior_bounds = [priors.distributions[k][1:] for k in model_info_to_include]
# --------------- TESTING -------------------
## P-P plot
#flow.pp_test(validation_dataset=pp_dataset,
#            parameter_labels=labels,
#            num_params=7)


# Generating results for some test data
samples, log_probabilities = flow.sample_and_logprob(test_dataset.tensors[1][0], # the conditional
                                                     num=1000) # number of samples we want to draw
samples = np.array([s[:,0] for s in samples]).T
result = FaultFlowResults(samples=samples,
                        conditional=[test_conditional[j][0] for j in range(len(test_conditional))],
                        log_probabilities=log_probabilities,
                        true_parameters=true_parameters,
                        parameter_labels=labels,
                        survey_coordinates=np.reshape(s.survey_coordinates, (s.shape[0], s.shape[1], 3))
                       )
plt.imshow(np.reshape(result.conditional[0], (50,50)))
plt.savefig('/data/www.astro/2263373r/survey.png')
plt.close()
result.directory = os.path.join(flow_location, f"testcase_preset_{0}/")
# plotting the surveys we are inverting
s.plot_contours(filename=os.path.join(result.directory, "survey_contours.png"), include_noise=True)
#dt_test.surveys[i].plot_pixels(filename=os.path.join(result.directory, "survey.png"))


# CORNER PLOTS                              
result.corner_plot(filename="corner_plot.png")
result.corner_plot(filename="corner_plot_prior.png", prior_bounds=prior_bounds)

result.plot_compare_surveys_samples(model_framework=model_framework, filename='survey_compare_samples.png', noise_scale=0.025, priors=priors)

result.plot_compare_surveys(model_framework=model_framework, filename='survey_compare.png', units='mGal', window=('tukey', 0.1), priors=priors, noise_scale=0.025)


