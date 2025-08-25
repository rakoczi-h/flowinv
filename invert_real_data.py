import os
import pickle as pkl
import pandas as pd
import numpy as np
import torch
import json
import sys
import matplotlib.pyplot as plt

from giflow.box import Box, BoxDataset
from giflow.survey import GravitySurvey
from giflow.flowmodel import FlowModel
from giflow.results import BoxFlowResults
from giflow.plot import compare_method_surveys

# Function
def rescale_bilby_samples(bilby_parameter_dict, parameters_to_rescale, scale_factor):
    for key in bilby_parameter_dict.keys():
        if key in parameters_to_rescale:
            bilby_parameter_dict[key]= [i * scale_factor for i in bilby_parameter_dict[key]]
    return bilby_parameter_dict


flow_location = '/data/www.astro/2263373r/giflow/4_paper/narrow_volume/voxelised_noisy/run_2024-09-28 21:31:40.559861/'
save_location = os.path.join(flow_location, 'qinetiq_data_gridordering_4_paper/')
if not os.path.exists(save_location):
    os.mkdir(save_location)
# -------------------- Reading in other results
with open("/scratch/balta1/2263373r/4_paper/real_bunker.pkl", 'rb') as file:
    dt_real = pkl.load(file)

truth = dt_real.boxes[0].voxelised_model # + np.random.normal(loc=0.0, scale=500, size=np.shape(dt_real.boxes[0].voxelised_model))
#dt_real.boxes[0].translate_to_parameterised_model()
#truth = dt_real.boxes[0].parameterised_model
#truth[:6] = truth[:6]*70

#units = ['[m]', '[m]', '[m]', '[m]', '[m]', '[m]', '' ]
# -------------------- Reading the flow --------------------------
device = torch.device('cuda')
flow=FlowModel()
flow.load(flow_location)
flow.flowmodel.to(device)
with open(os.path.join(flow.data_location, "validationset_0.pkl"), 'rb') as file:
    dt_val = pkl.load(file)
priors = dt_val.priors
labels = dt_val.parameter_labels
model_framework = dt_val.model_framework
survey_framework = dt_val.survey_framework
keys = dt_val.parameter_labels
prior_bounds = []
for k in keys:
    p = priors.distributions[k]
    if k != 'alpha':
        prior_bounds.append([p[1]*70, p[2]*70])
    else:
        prior_bounds.append([p[1], p[2]])

prior_samples = priors.sample(2000, returntype='array')
#print(dt_val.survey_framework)
# -------------------- Reading the data --------------------------
data_loc = '/scratch/balta1/2263373r/4_paper/qinetiq_data_4_paper.csv'
df = pd.read_csv(data_loc)

x = np.array(df['x'])
y = np.array(df['y'])
z = np.zeros(np.shape(x))

grav = -1*np.array(df['grav'])
grav = grav - np.min(grav)

noise_scale = 2.0149/70.0
print("Noise scale:", noise_scale)
print("Priors:", priors.distributions)

survey_coordinates = np.c_[x, y, z]


survey = GravitySurvey(ranges=dt_val.survey_framework['ranges'], survey_shape=dt_val.survey_framework['survey_shape'])
survey.make_survey()

reordered_coordinates = np.zeros(np.shape(survey_coordinates))
reordered_grav = np.zeros(np.shape(survey_coordinates)[0])
for i, s in enumerate(survey.survey_coordinates):
    diff = np.sqrt((survey_coordinates[:,0] - s[0])**2 + (survey_coordinates[:,1] - s[1])**2 + (survey_coordinates[:,2] - s[2])**2)
    reordered_coordinates[i,:] = survey_coordinates[np.argmin(diff),:]
    reordered_grav[i] = grav[np.argmin(diff)]

survey.gravity = reordered_grav-noise_scale/2

survey.noise_scale = noise_scale
survey.survey_coordinates = reordered_coordinates

data = {'x': survey.survey_coordinates[:,0], 'y': survey.survey_coordinates[:,1], 'grav': survey.gravity, 'noise_scale': survey.noise_scale}

df = pd.DataFrame(data=data)
df.to_csv(path_or_buf='/scratch/balta1/2263373r/4_paper/qinetiq_data_4_paper_inv_ready.csv')

survey.plot_contours(filename=os.path.join(save_location, 'survey.png'), include_noise=False)

dt_test = BoxDataset(size=1, priors=priors, survey_framework=dt_val.survey_framework, model_framework=dt_val.model_framework)
dt_test.surveys = [survey]

box = Box()
dt_test.boxes = [box]

survey_coordinates_to_include = ['x', 'y', 'noise_scale']
test_data, test_conditional = dt_test.make_data_for_network(survey_coordinates_to_include=survey_coordinates_to_include, model_info_to_include=[], add_noise=False, mix_survey_order=False)

test_conditional_tensor = flow.scalers['conditional'].scale_data(test_conditional)

test_conditional_tensor = torch.from_numpy(test_conditional_tensor.astype(np.float32)).to(device)

#test_dataset = flow.make_tensor_dataset(test_data, test_conditional, device=device, scale=True)
# --------------------- Results --------------------------
for i in range(1):
    samples, log_probabilities = flow.sample_and_logprob(test_conditional_tensor, num=100000)
    result = BoxFlowResults(samples=samples, conditional=[test_conditional[j][0] for j in range(len(test_conditional))],
        log_probabilities=log_probabilities,
        true_parameters=np.array([truth]),
        parameter_labels=[r'$c_x$', r'$c_y$', r'$c_z$', r'$l_x$', r'$l_y$', r'$l_z$', r'$\alpha$'],
        survey_coordinates=dt_test.surveys[0].survey_coordinates)
    result.directory = save_location
    #result.samples[:-1] = samples[:-1]*70
    #result.plot_voxel_volumes(model_framework=dt_test.model_framework, filename=f"3d_voxel_plot_{i}_v3.png")
    #result.corner_plot(filename="corner_plot_prior_bounds.png", prior_bounds=prior_bounds, units=units)
    #js = result.get_js_divergence(prior_samples)
    #print(js)
    #result.plot_compare_surveys(model_framework=dt_test.model_framework, filename="compare_survey.png", include_examples=True)

    #result.plot_compare_voxel_slices_pygimli(li_result, filename=f"compare_voxel_slices_{i}.png", normalisation=[-1000.0, 0.0], slice_coords=[[0,1,3], [7,8,9], [1,4,8]], plot_truth=True, model_framework=dt_test.model_framework)
    #result.plot_compare_voxel_slices(filename=f"compare_voxel_slices_{i}_v3.png", slice_coords=[[0,1,3], [7,8,9], [1,4,8]], plot_truth=True, normalisation=[-1000.0, 500.0], model_framework=dt_test.model_framework, aspect=[1.0, 0.5, 0.5], filter_noise=True)

    #result.plot_3D_statistics(model_framework=dt_test.model_framework, filename=f"3D_statistics_{i}.html")
    compare_method_surveys([result], [model_framework], [survey_framework], num=100, filename='/data/www.astro/2263373r/giflow/4_paper/survey_comparison_plot_4_paper_real.png')
