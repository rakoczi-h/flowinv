import os
import pickle as pkl
import pandas as pd
import numpy as np
import torch
import json

from giflow.box import Box, BoxDataset
from giflow.survey import GravitySurvey
from giflow.flowmodel import FlowModel
from giflow.results import BoxFlowResults
flow_location = '/data/www.astro/2263373r/giflow/box/parameterised/normalised/run_2024-06-03 10:13:21.656567/'
save_location = os.path.join(flow_location, 'qinetiq_data/')
if not os.path.exists(save_location):
    os.mkdir(save_location)
# -------------------- Reading the flow --------------------------
device = torch.device('cuda')
flow=FlowModel()
flow.load(flow_location)
flow.flowmodel.to(device)

with open(os.path.join(flow.data_location, "validationset.pkl"), 'rb') as file:
    dt_val = pkl.load(file)
priors = dt_val.priors
model_framework = dt_val.model_framework
print(dt_val.survey_framework)
# -------------------- Reading the data --------------------------
data_loc = '/data/wiay/2263373r/giflow/box/qinetiq_data.csv'
df = pd.read_csv(data_loc)

x = np.array(df['x'])
print(np.min(x))
print(np.min(x))
y = np.array(df['y'])
z = np.zeros(np.shape(x))
print(df['grav'])
grav = -1*np.array(df['grav'])*1000
grav = grav - np.min(grav)

width_real = np.max(x)-np.min(x)
width_train = np.max(dt_val.surveys[0].survey_coordinates[:,0])-np.min(dt_val.surveys[0].survey_coordinates[:,0])
scale_factor = width_real/width_train

survey_coordinates = np.c_[x/scale_factor, y/scale_factor, z/scale_factor]
print(np.max(dt_val.surveys[0].survey_coordinates))
print(np.min(dt_val.surveys[0].survey_coordinates))
survey = GravitySurvey(ranges=dt_val.survey_framework['ranges'], survey_shape=dt_val.survey_framework['survey_shape'], survey_coordinates=survey_coordinates)
survey.gravity = grav/scale_factor
box = Box()

dt_test = BoxDataset(size=1, priors=priors, survey_framework=dt_val.survey_framework, model_framework=dt_val.model_framework)
dt_test.surveys = [survey]
dt_test.boxes = [box]

dt_test.surveys[0].plot_pixels(filename=os.path.join(flow_location, f"qinetiq_data/survey.png"))
survey_coordinates_to_include = []
test_data, test_conditional = dt_test.make_data_arrays(survey_coordinates_to_include=survey_coordinates_to_include, include_noise=False)
test_conditional_tensor = flow.scalers['conditional'].transform(test_conditional.reshape(-1,test_conditional.shape[-1]))
test_conditional_tensor = test_conditional_tensor.reshape(1,-1)
test_conditional_tensor = torch.from_numpy(test_conditional_tensor.astype(np.float32)).to(device)
# --------------------- Results --------------------------
samples, log_probabilities = flow.sample_and_logprob(test_conditional_tensor, num=2000)

# Translating into voxelised model 
#samples_translated = []
#for s in samples:
#    parameters = dict.fromkeys(dt_val.parameter_labels)
#    for k, key in enumerate(dt_val.parameter_labels):
#        parameters[key] = s[k]
#    box = Box(parameters=parameters)
#    box.make_voxel_grid(grid_shape=model_framework['grid_shape'], ranges=model_framework['ranges'])
#    box.translate_to_voxels(background_noise_scale=model_framework['noise_scale'], density=model_framework['density'])
#    s_tr = box.voxelised_model
#    samples_translated.append(s_tr)
#samples_translated = np.array(samples_translated)
#
#result = BoxFlowResults(samples=samples_translated, conditional=test_conditional[0,:], log_probabilities=log_probabilities, survey_coordinates=dt_test.surveys[0].survey_coordinates)
#result.directory = save_location
#result.plot_compare_voxel_slices(filename='compare_voxel_slices.png', plot_truth=False)

result = BoxFlowResults(samples=samples, conditional=test_conditional[0,:], log_probabilities=log_probabilities, parameter_labels=dt_val.parameter_labels, survey_coordinates=dt_test.surveys[0].survey_coordinates)

#result = BoxFlowResults(samples=samples, conditional=test_conditional[0,:], log_probabilities=log_probabilities, parameter_labels=None, survey_coordinates=dt_test.surveys[0].survey_coordinates)
result.directory = save_location
#result.rescale(scaling_factor=scale_factor, parameters_to_rescale=['px', 'py', 'pz', 'lx', 'ly', 'lz'])

result.corner_plot(filename="corner_plot.png")
result.plot_compare_surveys(model_framework=dt_test.model_framework, filename="compare_survey.png", include_examples=True)

# Comparing to bilby
keys = dt_val.parameter_labels
with open('/data/www.astro/2263373r/giflow/bilby/box/standrews/inversion_result.json', 'r') as file:
    bilby_results = json.load(file)
    bilby_posterior_dict = bilby_results['posterior']['content']
    bilby_samples = []
    for key in keys:
        bilby_samples.append(bilby_posterior_dict[key])
    bilby_samples = np.array(bilby_samples).T
result.overlaid_corner(bilby_samples, ['Flow', 'Dynesty'], filename='overlaid_corner_bilby.png')
