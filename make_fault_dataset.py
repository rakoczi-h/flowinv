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
n = int(sys.argv[1])

# Specifying directories
save = '/scratch/balta0/2263373r/fault_python/synthetic_inversion_v2/'
if not os.path.exists(save):
    os.mkdir(save)

# Priors
distributions = {'cx': ['Uniform', -1.0, 1.0],
                 'cy': ['Uniform', -1.0, 1.0],
                 'l': ['Uniform', 1.0, 2.0],
                 'cz': ['Uniform', 0.1, 0.2],
                 'alpha': ['Uniform', 0.0, 2*np.pi]
                 #'density': ['Uniform', 500.0, 1000.0],
                 #'DL_ratio': ['Uniform', 0.01, 0.05]
                 }
priors = Prior(distributions=distributions)

# Defining the source model framework
model_framework = {
    "type": 'parameterised',
    'default_parameters': {'dip': 70*np.pi/180, 'density': 800.0, 'DL_ratio': 0.02},
    'varied_parameters': ['cx', 'cy', 'cz', 'l', 'alpha']
}

# Defining the gravimetry survey framework
survey_framework = {
    "shape": [50,50],
    "ranges": [[-2.0, 2.0],[-2.0, 2.0],[0]],
    "noise_scale": None,
    "noise_on_location_scale": 0.0,
    #"width_ratio" : ['Uniform', 0.1, 1.0],
    "width_ratio": None
}


# Training data
size = 10000 # Only making a small batch here, in reality we will likely need more data than this.
dt_train = FaultDataset(
   priors = priors,
   size = size,
   survey_framework = survey_framework,
   model_framework = model_framework
)
dt_train.make_dataset(augment=False)


filename = os.path.join(save, f"trainset_{n}.pkl")
with open(filename, 'wb') as file:
    pkl.dump(dt_train, file)

print(f"Memory use in gb: ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6+resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1e6)
print(f"Time taken: {datetime.now()-start_time}")


# Validation data
size = 1000 # Only making a small batch here, in reality we will likely need more data than this.
dt_train = FaultDataset(
   priors = priors,
   size = size,
   survey_framework = survey_framework,
   model_framework = model_framework
)
dt_train.make_dataset(augment=False)


filename = os.path.join(save, f"validationset_{n}.pkl")
with open(filename, 'wb') as file:
    pkl.dump(dt_train, file)

## PP data
#size = 100 # Only making a small batch here, in reality we will likely need more data than this.
#dt_train = FaultDataset(
#   priors = priors,
#   size = size,
#   survey_framework = survey_framework,
#   model_framework = model_framework
#)
#dt_train.make_dataset(augment=False)
#
#
#filename = os.path.join(save, f"ppset_{n}.pkl")
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
#dt_train.make_dataset(augment=False)
#
#
#filename = os.path.join(save, f"testset_{n}.pkl")
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
## Test data
#size = 6
#dt_train = FaultDataset(
#    priors = priors,
#    size = size,
#    survey_framework = survey_framework,
#    model_framework = model_framework
#)
#parameters_dict = {'cx': [0.0, 0.0, 0.5, 0.5, 0.0, 0.0], 'cy': [0.0, 0.0, 0.0, 0.0, 0.5, 0.5], 'l': [4.0, 4.0, 3.0, 3.0, 2.0, 2.0], 'alpha': [0.0, 350*np.pi/180, np.pi/2, np.pi/2, 0.0, 0.0], 'cz': [0.1, 0.1, 0.2, 0.2, 0.05, 0.05], 'DL_ratio': [0.02, 0.02, 0.02, 0.02, 0.02, 0.02], 'density': [800.0, 800.0, 800.0, 600.0, 600.0, 600.0]}
#dt_train.make_dataset(parameters_dict=parameters_dict, augment=False)
#
#filename = os.path.join(save, f"testset_1.pkl")
#with open(filename, 'wb') as file:
#    pkl.dump(dt_train, file)
#
#print(f"Dataset made. Time taken: {datetime.now()-start_time}")
#
#
