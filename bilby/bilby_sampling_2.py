#!/scratch/balta0/2263373r/conda_envs/giflow/bin/python
import bilby
import h5py
import json
import numpy as np
from datetime import datetime
import os
import shutil
import sys
import pickle as pkl
import pandas as pd

from giflow.box import Box
from giflow.survey import GravitySurvey

n = int(sys.argv[1])

label = "inversion"
bilby_outdir = "/data/www.astro/2263373r/giflow/4_paper/bilby/testcases_to_present_v2/"
bilby.utils.check_directory_exists_and_if_not_mkdir(bilby_outdir)

outdir = os.path.join(bilby_outdir, f"testcase_{n}/")
bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)

data_loc = '/scratch/balta0/2263373r/giflow/4_paper/parameterised/'

# ----------------------- Functions -------------------------------
def model(survey_coordinates, px, py, pz, lx, ly, lz, alpha):
    """
    Function defining the forward model.
    """
    start = datetime.now()
    box = Box(parameters = {"px": px, "py": py, "pz": pz, "lx": lx, "ly": ly, "lz": lz, "alpha": alpha})
    box.density = -2670.0 # density contrast
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
with open(os.path.join(data_loc, "testset_to_present_0.pkl"), 'rb') as file:
    dt_test = pkl.load(file)
box = dt_test.boxes[n]
survey = dt_test.surveys[n]
keys = dt_test.parameter_labels

survey.plot_pixels(filename=os.path.join(outdir, "survey.png"), include_noise=True)

data = survey.gravity+survey.noise
sigma = survey.noise_scale
print(sigma)
survey_coordinates = survey.survey_coordinates
# --------------------- Defining sampler inputs ------------------

# PRIOR
priors = prior(dt_test.priors.keys, dt_test.priors.distributions)
print(dt_test.priors.distributions)
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
