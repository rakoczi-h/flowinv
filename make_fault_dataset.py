import os
import sys
import resource
import pickle as pkl
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from giflow.dataset import FaultDataset
from giflow.prior import Prior

start_time = datetime.now()
# Specifying directories



n = int(sys.argv[1])

save = '/scratch/balta0/2263373r/fault_python/real_inversion/'

# Priors
distributions = {'cx': ['Uniform', -1.0, 1.0],
                 'cy': ['Uniform', -1.0, 1.0],
                 'l': ['Uniform', 1.0, 4.0],
                 'cz': ['Uniform', 0.01, 0.2],
                 'alpha': ['Uniform', 0.0, 2*np.pi],
                 'density': ['Uniform', 500.0, 1000.0],
                 'DL_ratio': ['Uniform', 0.01, 0.05]
                 }
priors = Prior(distributions=distributions)

# Defining the source model framework
model_framework = {
    "type": 'parameterised',
    'default_parameters': {'dip': 70*np.pi/180},
    'varied_parameters': ['cx', 'cy', 'cz', 'l', 'alpha', 'density', 'DL_ratio']
}

# Defining the gravimetry survey framework
survey_framework = {
    "shape": [50,50],
    "ranges": [[-0.5, 0.5],[-0.5, 0.5],[0]],
    "noise_scale": None,
    "noise_on_location_scale": 0.0,
    "width_ratio" : ['Uniform', 0.1, 1.0],
    #"width_ratio": None
}


## Training data
#size = 10000 # Only making a small batch here, in reality we will likely need more data than this.
#dt_train = FaultDataset(
#   priors = priors,
#   size = size,
#   survey_framework = survey_framework,
#   model_framework = model_framework
#)
#dt_train.make_dataset_v2(augment=True, augment_dims=['density'])
#
#
## print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6)
## print(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1e6)
#
#filename = os.path.join(save, f"trainset_{n}.pkl")
#with open(filename, 'wb') as file:
#    pkl.dump(dt_train, file)
#
##print(f"Time taken: {datetime.now()-start_time}")
#
## Validation data
#size = 1000 # Only making a small batch here, in reality we will likely need more data than this.
#dt_train = FaultDataset(
#   priors = priors,
#   size = size,
#   survey_framework = survey_framework,
#   model_framework = model_framework
#)
#dt_train.make_dataset_v2(augment=True, augment_dims=['density'])
#
#
#filename = os.path.join(save, f"validationset_{n}.pkl")
#with open(filename, 'wb') as file:
#    pkl.dump(dt_train, file)

# PP data
size = 100 # Only making a small batch here, in reality we will likely need more data than this.
dt_train = FaultDataset(
   priors = priors,
   size = size,
   survey_framework = survey_framework,
   model_framework = model_framework
)
dt_train.make_dataset_v2(augment=False)


filename = os.path.join(save, f"ppset_{n}.pkl")
with open(filename, 'wb') as file:
    pkl.dump(dt_train, file)


# Test data
size = 10 # Only making a small batch here, in reality we will likely need more data than this.
dt_train = FaultDataset(
   priors = priors,
   size = size,
   survey_framework = survey_framework,
   model_framework = model_framework
)
dt_train.make_dataset_v2(augment=False)


filename = os.path.join(save, f"testset_{n}.pkl")
with open(filename, 'wb') as file:
    pkl.dump(dt_train, file)

## Test data
#size = 1
#dt_train = FaultDataset(
#    priors = priors,
#    size = size,
#    survey_framework = survey_framework,
#    model_framework = model_framework
#)
#parameters_dict = {'cx': [0.25], 'cy': [0.5], 'l': [1.5], 'alpha': [np.pi/4], 'cz': [0.1]}
#dt_train.make_dataset(parameters_dict=parameters_dict)
#
#filename = os.path.join(save, f"testset_1.pkl")
#with open(filename, 'wb') as file:
#    pkl.dump(dt_train, file)
#
#print(f"Dataset made. Time taken: {datetime.now()-start_time}")


