#!/scratch/wiay/2263373r/masters/conda_envs/venv/bin/python

import pickle as pkl
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

from giflow.prior import Prior
from giflow.box import BoxDataset

n = int(sys.argv[1])
save_loc = '/data/wiay/2263373r/giflow/box/parameterised/normalised/'


# PRIOR
distributions = {"px": ['Uniform', -0.75, 0.75], "py": ['Uniform', -0.75, 0.75], "pz": ['Uniform', -1, -0.5],
    "lx": ['Uniform', 0, 1.5], "ly": ['Uniform', 0, 1.5], "lz": ['Uniform', 0, 1.0], "alpha": ['Uniform', 0, 1.5708]}
priors = Prior(distributions=distributions)

# FRAMEWORKS
survey_framework = {'noise_scale' : ['Uniform', 0.0, 0.25], 'survey_shape' : [8,8], 'ranges': [[-0.5,0.5],[-0.5, 0.5],[0]], 'noise_on_location_scale' : 0.0}
model_framework = {'type': 'parameterised', 'density': -1500.0, 'noise_scale': 50.0, 'grid_shape': [8,8,8], 'ranges': [[-0.75,0.75],[-0.75,0.75],[-1,0]]}

# Make train data:
size = 5000000
dt_train = BoxDataset(priors=priors, size=size, survey_framework=survey_framework, model_framework=model_framework)
dt_train.make_dataset()
file_name = os.path.join(save_loc, f"trainset_{n}.pkl")
with open(file_name, 'wb') as file:
    pkl.dump(dt_train, file)
print(f"Data set of size {size} made and saved as {file_name}.")


# Make validaton data:
#size = 10000
#dt_val = BoxDataset(priors=priors, size=size, survey_framework=survey_framework, model_framework=model_framework)
#dt_val.make_dataset()
#file_name = os.path.join(save_loc, 'validationset_v2.pkl')
#with open(file_name, 'wb') as file:
#    pkl.dump(dt_val, file)
#print(f"Data set of size {size} made and saved as {file_name}.")

## Make test data:
#size = 10
#dt_test = BoxDataset(priors=priors, size=size, survey_framework=survey_framework, model_framework=model_framework)
#parameters_dict = dict.fromkeys(priors.keys)
#parameters_dict['px'] = np.array([0, 0, 0, 0, 10, 10, 10, 10, 10, 10])
#parameters_dict['py'] = np.array([0, 0, 0, 0, 10, 10, 10, 10, 10, 10])
#parameters_dict['pz'] = np.array([-50, -70, -50, -70, -50, -70, -50, -70, -50, -70])
#parameters_dict['lx'] = np.array([90, 90, 80, 80, 60, 60, 70, 70, 40, 40])
#parameters_dict['ly'] = np.array([60, 60, 50, 50, 40, 40, 50, 50, 35, 35])
#parameters_dict['lz'] = np.array([60, 60, 60, 60, 60, 60, 60, 60, 60, 60])
#parameters_dict['alpha'] = np.array([0.8726, 0.8726, 0.8726, 0.8726, 0.8726, 0.8726, 0.8726, 0.8726, 0.8726, 0.8726])
#dt_test.make_dataset(parameters_dict=parameters_dict)
#file_name = os.path.join(save_loc, 'testset_v2.pkl')
#with open(file_name, 'wb') as file:
#    pkl.dump(dt_test, file)
#print(f"Data set of size {size} made and saved as {file_name}.")
