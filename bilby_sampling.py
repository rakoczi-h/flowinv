import bilby
import h5py
import json
import numpy as np
from datetime import datetime
import os
import shutil
import sys
import pickle as pkl

from giflow.box import Box

n = int(sys.argv[1]) # Give the index of the testcase as an input to the script

label = "inversion"
bilby_outdir = "/bilby_results/"

data_loc = '/data/'
data_name= 'testset.pkl'

# ----------------------- Functions -------------------------------
def model(survey_coordinates, px, py, pz, lx, ly, lz, alpha):
    """
    Function defining the forward model.
    """
    start = datetime.now()
    box = Box(parameters = {"px": px, "py": py, "pz": pz, "lx": lx, "ly": ly, "lz": lz, "alpha": alpha})
    box.density = -1500.0 # density contrast
    gz = box.forward_model(survey_coordinates.copy(), model_type='parameterised')
    end = datetime.now()
    return gz

def prior(keys, distributions):
    """
    Bilby prior function.
    """
    priors = dict.fromkeys(keys)
    for i, key in enumerate(keys):
        if distributions[key][0] == "Uniform":
            priors[key] = bilby.prior.Uniform(distributions[key][1], distributions[key][2], key)
        elif distributions[key][0] == "Normal":
            priors[key] = bilby.prior.Gaussian(distributions[key][1], distributions[key][2], key)
        elif isinstance(distributions[key][0], float) or isinstance(distributions[key][0], int) and len(distributions[key]) == 1:
            priors.pop(key, None)
    return priors

# --------------------- Reading data -----------------------------
with open(os.path.join(data_loc, data_name), 'rb') as file:
    dt = pkl.load(file)
box = dt.boxes[n]
survey = dt.surveys[n]
survey_coordinates = survey.survey_coordinates
data = survey.gravity + survey.noise
sigma = dt.survey_framework['noise_scale']
keys = box.parameter_labels

outdir = bilby_outdir + f"testcase_{n}/"
bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)

# --------------------- Defining sampler inputs ------------------

# PRIOR
priors = prior(dt.priors.keys, dt.priors.distributions)
# TRUTH
injection_parameters = dict.fromkeys(keys)
for idx, k in enumerate(keys):
     injection_parameters[k] = box.parameters[k]
# LIKELIHOOD
likelihood = bilby.likelihood.GaussianLikelihood(survey_coordinates, data, model, sigma)

# -------------------- Running sampler ---------------------------
result = bilby.run_sampler(
   likelihood=likelihood,
   priors=priors,
   sampler="dynesty",
   nlive=1000,
   maxmcmc = 10000,
   injection_parameters=injection_parameters,
   outdir=outdir,
   label=label,
)
result.plot_corner()
