#!/scratch/wiay/2263373r/masters/conda_envs/flowenv/bin/python

import pickle as pkl
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

from giflow.prior import Prior
from giflow.box import BoxDataset

n = int(sys.argv[1])
save_loc = '/scratch/balta0/2263373r/giflow/box/voxelised/noisy_grid/'
if not os.path.exists(save_loc):
    os.mkdir(save_loc)


# PRIOR
distributions = {"px": ['Uniform', -0.75, 0.75], "py": ['Uniform', -0.75, 0.75], "pz": ['Uniform', -0.75, 0.0],
    "lx": ['Uniform', 0, 1.5], "ly": ['Uniform', 0, 1.5], "lz": ['Uniform', 0, 0.75], "alpha": ['Uniform', 0, 1.5708]}
priors = Prior(distributions=distributions)

# FRAMEWORKS
survey_framework = {'noise_scale' : ['Uniform', 0.0, 0.25], 'survey_shape' : [8,8], 'ranges': [[-0.5,0.5],[-0.5, 0.5],[0]], 'noise_on_location_scale' : 0.05}
model_framework = {'type': 'voxelised', 'density': -2670.0, 'noise_scale': 500.0, 'grid_shape': [8,8,8], 'ranges': [[-0.75,0.75],[-0.75,0.75],[-1.5,0]]}


# Make train data:
size = 500000
dt_train = BoxDataset(priors=priors, size=size, survey_framework=survey_framework, model_framework=model_framework)
dt_train.make_dataset()
file_name = os.path.join(save_loc, f"trainset_{n}.pkl")
with open(file_name, 'wb') as file:
    pkl.dump(dt_train, file)
print(f"Data set of size {size} made and saved as {file_name}.")


# Make validaton data:
size = 100000
dt_val = BoxDataset(priors=priors, size=size, survey_framework=survey_framework, model_framework=model_framework)
dt_val.make_dataset()
file_name = os.path.join(save_loc, 'validationset_0.pkl')
with open(file_name, 'wb') as file:
    pkl.dump(dt_val, file)
print(f"Data set of size {size} made and saved as {file_name}.")


## Make test data:
#size = 10
#dt_test = BoxDataset(priors=priors, size=size, survey_framework=survey_framework, model_framework=model_framework)
#parameters_dict = dict.fromkeys(priors.keys)
#parameters_dict['px'] = np.array([0, 0, 0, 0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
#parameters_dict['py'] = np.array([0, 0, 0, 0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
#parameters_dict['pz'] = np.array([-0.55, -0.75, -0.55, -0.75, -0.55, -0.75, -0.55, 0.0, 0.0, 0.0])
#parameters_dict['lx'] = np.array([1, 1, 0.8, 0.8, 0.6, 0.6, 0.7, 0.7, 0.4, 0.4])
#parameters_dict['ly'] = np.array([0.6, 0.6, 0.6, 0.6, 0.4, 0.4, 0.5, 0.5, 0.35, 0.35])
#parameters_dict['lz'] = np.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6])
#parameters_dict['alpha'] = np.array([0.8726, 0.8726, 0.8726, 0.8726, 0.8726, 0.8726, 0.8726, 0.8726, 0.8726, 0.8726])
#dt_test.make_dataset(parameters_dict=parameters_dict)
#file_name = os.path.join(save_loc, 'testset_0.pkl')
#with open(file_name, 'wb') as file:
#    pkl.dump(dt_test, file)
#print(f"Data set of size {size} made and saved as {file_name}.")

#size = 9
#dt_test = BoxDataset(priors=priors, size=size, survey_framework=survey_framework, model_framework=model_framework)
#parameters_dict = dict.fromkeys(priors.keys)
#parameters_dict['px'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
#parameters_dict['py'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
#parameters_dict['pz'] = np.array([-0.55, -0.55, -0.55, -0.55, -0.55, -0.55, -0.55, -0.55, -0.55])
#parameters_dict['lx'] = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
#parameters_dict['ly'] = np.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6])
#parameters_dict['lz'] = np.array([0.2, 0.4, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0])
#parameters_dict['alpha'] = np.array([0.8726, 0.8726, 0.8726, 0.8726, 0.8726, 0.8726, 0.8726, 0.8726, 0.8726])
#dt_test.make_dataset(parameters_dict=parameters_dict)
#
#for i, s in enumerate(dt_test.surveys):
#    s.plot_pixels(filename=f"/data/www.astro/2263373r/survey_{i}.png", include_noise=True)
