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
from giflow.prior import Prior
from giflow.plot import make_gif


label = "inversion"
bilby_outdir = "/data/www.astro/2263373r/giflow/bilby/box/qinetiq_data_half_v2/"
bilby.utils.check_directory_exists_and_if_not_mkdir(bilby_outdir)

data_loc = '/scratch/balta0/2263373r/giflow/box/'
data_name= 'qinetiq_data_v5.csv'

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

def plot_posterior_samples(result, parameter_keys, save_location):
    posterior_df = result.posterior
    prior = result.priors
    log_likelihood = np.array(posterior_df['log_likelihood'])
    num_samples = 200
    indices = np.flip(np.argsort(log_likelihood))[:num_samples]
    box_parameters = dict.fromkeys(parameter_keys)
    image_names = []
    for n in indices:
        for key in parameter_keys:
            box_parameters[key] = posterior_df[key][n]
        box = Box(parameters=box_parameters, density=-2670, parameter_labels=parameter_keys)
        image_name = f"box_{n}.png"
        image_names.append(image_name)
        box.intact_plot_3D_mesh(filename=os.path.join(save_location, image_name), axis_limits=np.array([[prior['px'].minimum, prior['px'].maximum],[prior['py'].minimum, prior['py'].maximum],[prior['pz'].minimum-prior['lz'].maximum, prior['pz'].maximum]]))
    make_gif(image_names, save_location, os.path.join(save_location, "3D_samples.gif"))
    for image_name in image_names:
        os.remove(os.path.join(save_location, image_name))

# --------------------- Reading data -----------------------------
# Reading data set that was used for flow inversion, to ensure that the same priors are used
#with open(os.path.join('/scratch/balta0/2263373r/giflow/box/parameterised/single_noise_level/validationset_0.pkl'), 'rb') as file:
#    dt_val = pkl.load(file)
#keys = dt_val.parameter_labels

df = pd.read_csv(os.path.join(data_loc, data_name))

x = np.array(df['x'])
dx = np.max(x)-np.min(x)
#x = x - dx/2
y = np.array(df['y'])
dy = np.max(y)-np.min(y)
#y = y - dy/2
z = np.zeros(np.shape(x))

grav = -1*np.array(df['grav'])
grav = grav - np.min(grav)

noise_scale = 4.7357

survey_coordinates = np.c_[x, y, z]

data = grav
sigma = noise_scale

#width_real = np.max(x)-np.min(x)
#width_train = np.max(dt_val.surveys[0].survey_coordinates[:,0])-np.min(dt_val.surveys[0].survey_coordinates[:,0])
#scale_factor = width_real/width_train



#survey_coordinates = np.c_[x/scale_factor, y/scale_factor, z/scale_factor]

survey = GravitySurvey(ranges=[[-dx/2, dx/2],[-dy/2, dy/2],[0]], survey_shape=[167], survey_coordinates=survey_coordinates)
survey.gravity = grav
#survey.noise_scale = noise_scale/scale_factor
#data = survey.gravity
#sigma =  survey.noise_scale # this comes from looking at the original survey data standard deviations

survey.plot_contours(filename=os.path.join(bilby_outdir, 'survey.png'), include_noise=False)

outdir = bilby_outdir
bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)

# --------------------- Defining sampler inputs ------------------
distributions = {"px": ['Uniform', -100, 100], "py": ['Uniform', -100, 100], "pz": ['Uniform', -70.0, 0.0],
    "lx": ['Uniform', 0, 200], "ly": ['Uniform', 0, 200], "lz": ['Uniform', 0, 100], "alpha": ['Uniform', 0, 1.5708]}
priors = Prior(distributions=distributions)
# PRIOR
priors = prior(priors.keys, distributions)

# TRUTH
#injection_parameters = dict.fromkeys(keys)
#for idx, k in enumerate(keys):
#     injection_parameters[k] = box.parameters[k]
# LIKELIHOOD
likelihood = bilby.likelihood.GaussianLikelihood(survey_coordinates, data, model, sigma)

# -------------------- Running sampler ---------------------------
result = bilby.run_sampler(
   likelihood=likelihood,
   priors=priors,
   sampler="dynesty",
   nlive=1000,
   maxmcmc = 10000,
   #injection_parameters=injection_parameters,
   outdir=outdir,
   label=label,
)
result.plot_corner()

sample_plots_dir = os.path.join(bilby_outdir, 'samples/')
if not os.path.exists(sample_plots_dir):
    os.mkdir(sample_plots_dir)

plot_posterior_samples(result, priors.keys(), sample_plots_dir)
