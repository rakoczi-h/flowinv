import os
import sys
import resource
import pickle as pkl
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import json

from giflow.dataset import FaultDataset
from giflow.prior import Prior

start_time = datetime.now()
n = int(sys.argv[1])

# Specifying directories
save = '/scratch/balta0/2263373r/fault_python/inversion_dataset_12_dim_v2/'
if not os.path.exists(save):
    os.mkdir(save)

# Priors
distributions = {'cx': ['Uniform', -1.0, 1.0],
                 'cy': ['Uniform', -1.0, 1.0],
                 'l': ['Uniform', 1.0, 4.0],
                 'cz': ['Uniform', 0.01, 0.2],
                 'alpha': ['Uniform', 0.0, 2*np.pi],
                 'density': ['Uniform', 500.0, 1000.0],
                 'DL_ratio': ['Uniform', 0.01, 0.05],
                 'dip': ['Uniform', 30*np.pi/180, 80*np.pi/180],
                 'Blend_order': ['Uniform', 0.3, 1.8],
                  'Displacement_order': ['Uniform', 1.2, 2.4],
                  'Extent_ratio': ['Uniform', 0.5, 2.0],
                    'sym_factor': ['Uniform', 0.1, 0.5]
                 }
priors = Prior(distributions=distributions)

with open(os.path.join(save, 'prior.pkl'), 'wb') as f:
    pkl.dump(priors, f)

# Defining the source model framework
model_framework = {
    "type": 'parameterised',
    'default_parameters': {},
    'varied_parameters': ['cx', 'cy', 'cz', 'l', 'alpha', 'density', 'DL_ratio', 'dip', 'Blend_order', 'Displacement_order', 'Extent_ratio', 'sym_factor']
 }
with open(os.path.join(save, 'model_framework.json'), 'w') as f:
    json.dump(model_framework, f)

# Defining the gravimetry survey framework
survey_framework = {
    "shape": [50,50],
    "ranges": [[-0.5, 0.5],[-0.5, 0.5],[0]],
    "noise_scale": None,
    "noise_on_location_scale": 0.0,
    "width_ratio" : ['Uniform', 0.1, 1.0],
    #"width_ratio": None
}

with open(os.path.join(save, 'survey_framework.json'), 'w') as f:
    json.dump(survey_framework, f)

##Training data
#size = 10000 # Only making a small batch here, in reality we will likely need more data than this.
#dt_train = FaultDataset(
#  priors = priors,
#  size = size,
#  survey_framework = survey_framework,
#  model_framework = model_framework
#)
#dt_train.make_dataset(augment=False, augment_dims=['density'], augment_num=10, window=True, zero_pad=True, num_components=100)
#
#
#filename = os.path.join(save, f"trainset_{n}.pkl")
#with open(filename, 'wb') as file:
#   pkl.dump(dt_train, file)
#
#print(f"Memory use in gb: ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6+resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1e6)
#print(f"Time taken: {datetime.now()-start_time}")


## Validation data
#size = 1000 # Only making a small batch here, in reality we will likely need more data than this.
#dt_train = FaultDataset(
#  priors = priors,
#  size = size,
#  survey_framework = survey_framework,
#  model_framework = model_framework
#)
#dt_train.make_dataset(augment=False, augment_dims=['density'], augment_num=10, window=True, zero_pad=True, num_components=100)
#
#
#filename = os.path.join(save, f"validationset_{n}.pkl")
#with open(filename, 'wb') as file:
#    pkl.dump(dt_train, file)

## PP data
#size = 100 # Only making a small batch here, in reality we will likely need more data than this.
#dt_train = FaultDataset(
#   priors = priors,
#   size = size,
#   survey_framework = survey_framework,
#   model_framework = model_framework
#)
#dt_train.make_dataset(augment=False, window=True, zero_pad=True, num_components=100)
#
#
#filename = os.path.join(save, f"ppset_0.pkl")
#with open(filename, 'wb') as file:
#    pkl.dump(dt_train, file)
#
#
## Test data
#size = 10 # Only making a small batch here, in reality we will likely need more data than this.
#dt_train = FaultDataset(
#   priors = priors,
#   size = size,
#   survey_framework = survey_framework,
#   model_framework = model_framework
#)
#dt_train.make_dataset(augment=False, window=True, zero_pad=True, num_components=100)
#
#
#filename = os.path.join(save, f"testset_1.pkl")
#with open(filename, 'wb') as file:
#    pkl.dump(dt_train, file)


#survey_framework = {
#    "shape": [50,50],
#    "ranges": [[-0.5, 0.5],[-0.1, 0.1],[0]],
#    "noise_scale": None,
#    "noise_on_location_scale": 0.0,
#    #"width_ratio" : ['Uniform', 0.1, 1.0],
#    "width_ratio": None
#}
# Test data

survey_framework = {
    "shape": [50,50],
    "ranges": [[-0.5, 0.5],[-0.5, 0.5],[0]],
    "noise_scale": None,
    "noise_on_location_scale": 0.0,
    "width_ratio" : [1.0],
    #"width_ratio": None
}

size = 1
dt_train = FaultDataset(
    priors = priors,
    size = size,
    survey_framework = survey_framework,
    model_framework = model_framework
)
F = 4
parameters_dict = {'cx': [0.25/4], 'cy': [0.5/4], 'l': [1.5/4], 'alpha': [np.pi/4], 'cz': [0.1/4], 'DL_ratio': [0.02], 'Extent_ratio': [1.5], 'sym_factor': [0.2], 'Blend_order': [1.2], 'Displacement_order': [1.2], 'density': [800.0], 'dip': [70*np.pi/180]}
dt_train.make_dataset(parameters_dict=parameters_dict, augment=False, augment_dims=['density'], augment_num=10, window=True, zero_pad=True, num_components=100)


filename = os.path.join(save, f"testset_2.pkl")
with open(filename, 'wb') as file:
    pkl.dump(dt_train, file)

print(f"Dataset made. Time taken: {datetime.now()-start_time}")
#
#
