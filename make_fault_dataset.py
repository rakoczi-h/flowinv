import os
import sys
import pickle as pkl
import numpy as np
from datetime import datetime

from giflow.dataset import FaultDataset
from giflow.prior import Prior

start_time = datetime.now()
# Specifying directories
<<<<<<< HEAD
save = '/scratch/balta0/2263373r/fault_python/'
if not os.path.exists(save):
    os.mkdir(save)

n = int(sys.argv[1])
=======
save = './test_dataset/'
if not os.path.exists(save):
    os.mkdir(save)

>>>>>>> ee3f09e (Made make_fault_dataset.py file)

# Priors
distributions = {'cx': ['Uniform', -1.0, 1.0],
                 'cy': ['Uniform', -1.0, 1.0],
<<<<<<< HEAD
                 'l': ['Uniform', 1.0, 2.0],
=======
                 'l': ['Uniform', 1.0, 20.0],
>>>>>>> ee3f09e (Made make_fault_dataset.py file)
                 'cz': ['Uniform', 0.01, 0.2],
                 'alpha': ['Uniform', 0.0, 2*np.pi]}
priors = Prior(distributions=distributions)

# Defining the source model framework
model_framework = {
<<<<<<< HEAD
    "type": 'parameterised',
    "density": 800.0,
    "shape": [50,50],
    "ranges": [[-2.0, 2.0], [-2.0, 2.0], [0.0]],
    'default_parameters': {'DL_ratio': 0.02, 'dip': 70*np.pi/180}
=======
    'default_parameters': {'DL_ratio': 0.02, 'dip': 70*np.pi/180, 'density': 800.0},
    'varied_parameters': ['cx', 'cy', 'cz', 'l', 'alpha']
>>>>>>> ee3f09e (Made make_fault_dataset.py file)
}

# Defining the gravimetry survey framework
survey_framework = {
    "shape": [50,50],
    "ranges": [[-2, 2],[-2, 2],[0]],
    "noise_scale": 0.1,
    "noise_on_location_scale": 0.0,
    "randomise_centre" : False,
<<<<<<< HEAD
    "width_ratio" : None
=======
    "width_ratio" : ['Uniform', 0.1, 0.5]
>>>>>>> ee3f09e (Made make_fault_dataset.py file)
}



<<<<<<< HEAD
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
=======
# Training data
size = 10 # Only making a small batch here, in reality we will likely need more data than this.
dt_train = FaultDataset(
   priors = priors,
   size = size,
   survey_framework = survey_framework,
   model_framework = model_framework
)
dt_train.make_dataset_v2()

# filename = os.path.join(save, f"trainset.pkl")
# with open(filename, 'wb') as file:
#    pkl.dump(dt_train, file)

print(f"Dataset made. Time taken: {datetime.now()-start_time}")

for i, f in enumerate(dt_train.sourcemodels):
    dt_train.surveys[i].plot_contours(filename=os.path.join(save, f"surveys_{i}.png"))
    f.plot_3D_surface(filename=os.path.join(save, f"fault_{i}.png"), survey_coordinates=dt_train.surveys[i].survey_coordinates)

>>>>>>> ee3f09e (Made make_fault_dataset.py file)
#
#
## Validation data
#size = 10000 # Only making a small batch here, in reality we will likely need more data than this.
#dt_train = FaultDataset(
#    priors = priors,
#    size = size,
#    survey_framework = survey_framework,
#    model_framework = model_framework
#)
#dt_train.make_dataset()
#
#filename = os.path.join(save, f"validationset_{n}.pkl")
#with open(filename, 'wb') as file:
#    pkl.dump(dt_train, file)
#
#print(f"Dataset made. Time taken: {datetime.now()-start_time}")

<<<<<<< HEAD
# PP data
size = 100 # Only making a small batch here, in reality we will likely need more data than this.
dt_train = FaultDataset(
    priors = priors,
    size = size,
    survey_framework = survey_framework,
    model_framework = model_framework
)
dt_train.make_dataset()

filename = os.path.join(save, f"ppset_{n}.pkl")
with open(filename, 'wb') as file:
    pkl.dump(dt_train, file)

print(f"Dataset made. Time taken: {datetime.now()-start_time}")

# PP data
size = 10 # Only making a small batch here, in reality we will likely need more data than this.
dt_train = FaultDataset(
    priors = priors,
    size = size,
    survey_framework = survey_framework,
    model_framework = model_framework
)
dt_train.make_dataset()

filename = os.path.join(save, f"testset_2.pkl")
with open(filename, 'wb') as file:
    pkl.dump(dt_train, file)

print(f"Dataset made. Time taken: {datetime.now()-start_time}")
=======
# # PP data
# size = 100 # Only making a small batch here, in reality we will likely need more data than this.
# dt_train = FaultDataset(
#     priors = priors,
#     size = size,
#     survey_framework = survey_framework,
#     model_framework = model_framework
# )
# dt_train.make_dataset()

# filename = os.path.join(save, f"ppset_{n}.pkl")
# with open(filename, 'wb') as file:
#     pkl.dump(dt_train, file)

# print(f"Dataset made. Time taken: {datetime.now()-start_time}")

# # PP data
# size = 10 # Only making a small batch here, in reality we will likely need more data than this.
# dt_train = FaultDataset(
#     priors = priors,
#     size = size,
#     survey_framework = survey_framework,
#     model_framework = model_framework
# )
# dt_train.make_dataset()

# filename = os.path.join(save, f"testset_2.pkl")
# with open(filename, 'wb') as file:
#     pkl.dump(dt_train, file)

# print(f"Dataset made. Time taken: {datetime.now()-start_time}")
>>>>>>> ee3f09e (Made make_fault_dataset.py file)

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


