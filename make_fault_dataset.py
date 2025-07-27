import os
import sys
import pickle as pkl
import numpy as np
from datetime import datetime

from giflow.dataset import FaultDataset
from giflow.prior import Prior

start_time = datetime.now()
# Specifying directories


# Priors
distributions = {'cx': ['Uniform', -1.0, 1.0],
                 'cy': ['Uniform', -1.0, 1.0],
                 'l': ['Uniform', 1.0, 2.0],
                 'cz': ['Uniform', 0.01, 0.2],
                 'alpha': ['Uniform', 0.0, 2*np.pi]}
priors = Prior(distributions=distributions)

# Defining the source model framework
model_framework = {
    "type": 'parameterised',
    "shape": [50,50],
    "ranges": [[-2.0, 2.0], [-2.0, 2.0], [0.0]],
    'default_parameters': {'DL_ratio': 0.02, 'dip': 70*np.pi/180}
    'default_parameters': {'DL_ratio': 0.02, 'dip': 70*np.pi/180, 'density': 800.0},
    'varied_parameters': ['cx', 'cy', 'cz', 'l', 'alpha']
}

# Defining the gravimetry survey framework
survey_framework = {
    "shape": [50,50],
    "ranges": [[-2, 2],[-2, 2],[0]],
    "noise_scale": 0.1,
    "noise_on_location_scale": 0.0,
    "randomise_centre" : False,
    "width_ratio" : None
}



## Training data
#size = 100000 # Only making a small batch here, in reality we will likely need more data than this.
#dt_train = FaultDataset(
#    priors = priors,
#    size = size,
#    survey_framework = survey_framework,
#    model_framework = model_framework
#)
#dt_train.make_dataset()
#
#filename = os.path.join(save, f"trainset_{n}.pkl")
#with open(filename, 'wb') as file:
#    pkl.dump(dt_train, file)
#
#print(f"Dataset made. Time taken: {datetime.now()-start_time}")




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


