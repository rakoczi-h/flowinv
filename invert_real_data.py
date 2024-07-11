import os
import pickle as pkl
import pandas as pd
import numpy as np
import torch
import json
import matplotlib.pyplot as plt

from giflow.box import Box, BoxDataset
from giflow.survey import GravitySurvey
from giflow.flowmodel import FlowModel
from giflow.results import BoxFlowResults


# Function
def rescale_bilby_samples(bilby_parameter_dict, parameters_to_rescale, scale_factor):
    for key in bilby_parameter_dict.keys():
        if key in parameters_to_rescale:
            bilby_parameter_dict[key]= [i * scale_factor for i in bilby_parameter_dict[key]]
    return bilby_parameter_dict



flow_location = '/data/www.astro/2263373r/giflow/box/voxelised/noisy_grid/run_2024-07-09 10:56:14.929649/'

save_location = os.path.join(flow_location, 'qinetiq_data/')
if not os.path.exists(save_location):
    os.mkdir(save_location)
# -------------------- Reading the flow --------------------------
device = torch.device('cuda')
flow=FlowModel()
flow.load(flow_location)
flow.flowmodel.to(device)
print(flow.data_location)
with open(os.path.join(flow.data_location, "validationset_0.pkl"), 'rb') as file:
    dt_val = pkl.load(file)
priors = dt_val.priors
model_framework = dt_val.model_framework

#print(dt_val.survey_framework)
# -------------------- Reading the data --------------------------
data_loc = '/scratch/balta0/2263373r/giflow/qinetiq_data.csv'
df = pd.read_csv(data_loc)

x = np.array(df['x'])
y = np.array(df['y'])
z = np.zeros(np.shape(x))

grav = -1*np.array(df['grav'])
grav = grav - np.min(grav)

#shuffling the points
i_arr = np.arange(grav.size)
np.random.shuffle(i_arr)
grav = grav[i_arr]
x = x[i_arr]
y = y[i_arr]
z = z[i_arr]

noise_scale = 4.7357/np.sqrt(4)

width_real = 70.0
width_train = np.max(dt_val.survey_framework['ranges'][0][1]-dt_val.survey_framework['ranges'][0][0])
scale_factor = width_real/width_train

print("Scale factor:", scale_factor)
print("Noise scale:", noise_scale)
print("Priors:", priors.distributions)

survey_coordinates = np.c_[x/scale_factor, y/scale_factor, z/scale_factor]

survey = GravitySurvey(ranges=dt_val.survey_framework['ranges'], survey_shape=dt_val.survey_framework['survey_shape'], survey_coordinates=survey_coordinates)
survey.gravity = grav/scale_factor
survey.noise_scale = noise_scale/scale_factor

survey.plot_contours(filename=os.path.join(save_location, 'survey.png'), include_noise=False)

dt_test = BoxDataset(size=1, priors=priors, survey_framework=dt_val.survey_framework, model_framework=dt_val.model_framework)
dt_test.surveys = [survey]

box = Box()
dt_test.boxes = [box]

survey_coordinates_to_include = ['x', 'y', 'noise_scale']
test_data, test_conditional = dt_test.make_data_arrays(survey_coordinates_to_include=survey_coordinates_to_include, add_noise=False)

test_conditional_tensor = flow.scalers['conditional'].scale_data(test_conditional)

test_conditional_tensor = torch.from_numpy(test_conditional_tensor.astype(np.float32)).to(device)

#test_dataset = flow.make_tensor_dataset(test_data, test_conditional, device=device, scale=True)
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

result = BoxFlowResults(samples=samples, conditional=[test_conditional[j][0] for j in range(len(test_conditional))], log_probabilities=log_probabilities, parameter_labels=dt_val.parameter_labels, survey_coordinates=dt_test.surveys[0].survey_coordinates)

#result = BoxFlowResults(samples=samples, conditional=test_conditional[0,:], log_probabilities=log_probabilities, parameter_labels=None, survey_coordinates=dt_test.surveys[0].survey_coordinates)
result.directory = save_location
#result.rescale(scaling_factor=scale_factor, parameters_to_rescale=['px', 'py', 'pz', 'lx', 'ly', 'lz'])

#result.corner_plot(filename="corner_plot.png")
#result.plot_compare_surveys(model_framework=dt_test.model_framework, filename="compare_survey.png", include_examples=True)


result.plot_compare_voxel_slices(filename=f"compare_voxel_slices.png", normalisation=[dt_val.boxes[0].density, 0.0], slice_coords=[2,5,8])


result.plot_3D_statistics(model_framework=dt_test.model_framework, axis_scale=scale_factor)


#result.plot_3D_samples(model_framework=dt_test.model_framework, num_to_plot=2000, mode='cumulativemean', filename='3D_cumulativemean.gif', axis_scale=scale_factor)
#result.plot_3D_samples(model_framework=dt_test.model_framework, num_to_plot=50, mode='maxlikelihood', filename='3D_samples.gif', axis_scale=scale_factor)



#rescale = False
#parameters_to_rescale = ['px', 'py', 'pz', 'lx', 'ly', 'lz']
#if rescale:
#    result.rescale(scale_factor=scale_factor, parameters_to_rescale=parameters_to_rescale)
## Comparing to bilby
#keys = dt_val.parameter_labels
#with open('/data/www.astro/2263373r/giflow/bilby/box/standrews_noise_realistic_v6/inversion_result.json', 'r') as file:
#    bilby_results = json.load(file)
#    bilby_posterior_dict = bilby_results['posterior']['content']
#    if rescale:
#        bilby_posterior_dict = rescale_bilby_samples(bilby_posterior_dict, parameters_to_rescale, scale_factor=scale_factor)
#    bilby_samples = []
#    for key in keys:
#        bilby_samples.append(bilby_posterior_dict[key])
#    bilby_samples = np.array(bilby_samples).T
#
#
#if rescale:
#    result.overlaid_corner(bilby_samples, ['Flow', 'Dynesty'], filename='overlaid_corner_bilby_rescaled.png')
#else:
#    result.overlaid_corner(bilby_samples, ['Flow', 'Dynesty'], filename='overlaid_corner_bilby.png')
