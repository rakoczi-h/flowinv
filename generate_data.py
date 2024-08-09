import os
import pickle as pkl

from giflow.box import BoxDataset
from giflow.prior import Prior

# Specifying directories
save = 'data'
if not os.path.exists(save):
    os.mkdir(save)

# Priors
distributions = {
    "px": ['Uniform', -0.75, 0.75],
    "py": ['Uniform', -0.75, 0.75],
    "pz": ['Uniform', -0.75, 0.0], # We do not want the box to extend over the surface!
    "lx": ['Uniform', 0.0, 1.5],
    "ly": ['Uniform', 0.0, 1.5],
    "lz": ['Uniform', 0.0, 0.75],
    "alpha": ['Uniform', 0.0, 1.5708]
}
priors = Prior(distributions=distributions)

priors.plot_distributions(filename=os.path.join(save, 'priors.png'))

# Defining the source model framework
model_framework = {
    "type": 'parameterised',
    "density": -2670.0
}

# Defining the gravimetry survey framework
survey_framework = {
    "survey_shape": [8,8],
    "ranges": [[-0.5, 0.5],[-0.5, 0.5],[0.0]],
    "noise_scale": ['Uniform', 0.0, 0.25],
    "noise_on_location_scale": 0.0
}

# Training data
size = 5000 # Only making a small batch here, in reality we will likely need more data than this.
dt_train = BoxDataset(
    priors = priors,
    size = size,
    survey_framework = survey_framework,
    model_framework = model_framework
)
dt_train.make_dataset()

filename = os.path.join(save, 'trainset.pkl')
with open(filename, 'wb') as file:
    pkl.dump(dt_train, file)

# Validation data
size = 500
dt_train = BoxDataset(
    priors = priors,
    size = size,
    survey_framework = survey_framework,
    model_framework = model_framework
)
dt_train.make_dataset()

filename = os.path.join(save, 'valset.pkl')
with open(filename, 'wb') as file:
    pkl.dump(dt_train, file)
