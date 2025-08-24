import bilby
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl

from giflow.utils import pad_grid
from giflow.fault import Fault
from giflow.survey import GravitySurvey


label = "5_parameter_V2"
bilby_outdir = 'outdir'
bilby.utils.check_directory_exists_and_if_not_mkdir(bilby_outdir)
bilby_outdir = os.path.join(bilby_outdir, label)
bilby.utils.check_directory_exists_and_if_not_mkdir(bilby_outdir)

# ----------------------- Functions -------------------------------
def model(survey_coordinates, cx, cy, l, alpha, cz):
    """
    Function defining the forward model.
    """
    fault = Fault(parameters = {"cx": cx, "cy": cy, "l": l, "alpha": alpha, "cz": cz,
                                 "density": 800.0,
                                 "DL_ratio": 0.02,
                                 "dip": 70*np.pi/180,
                                "Displacement_order": 1.2,
                                "Blend_order": 1.2,
                                "Extent_ratio": 1.5,
                                "sym_factor": 0.2})
    
    pad = int(np.shape(survey_coordinates)[0]*0.5) # padding the fault grid by 25%
    grid = pad_grid(survey_coordinates, pad, square=True)
    fault.make_fault(grid=grid)
    fault.displacement_profile[fault.displacement_profile>fault.parameters['cz']] = fault.parameters['cz']
    gz, _, _, _= fault.forward_model(survey_coordinates=survey_coordinates, num_components=50, win=('tukey', 0.1), remove_min=True, zero_pad=True, pad_width=[np.shape(grid)[0], np.shape(grid)[0]])

    return gz.flatten()

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

# --------------------- Defining sampler inputs ------------------
keys = ['cx', 'cy', 'l', 'alpha', 'cz']
sigma = 0.1

# SURVEY GRID
survey_shape = [50, 50]
ranges = [[-2.0, 2.0], [-2.0, 2.0], [0]]
X = np.linspace(ranges[0][0], ranges[0][1], num=survey_shape[0])
Y = np.linspace(ranges[1][0], ranges[1][1], num=survey_shape[1])
X, Y = np.meshgrid(X, Y, indexing='ij')
X = np.expand_dims(X, axis=2)
Y = np.expand_dims(Y, axis=2)
Z = np.zeros(np.shape(X))
survey_coordinates = np.c_[X, Y, Z]

# PRIOR
prior_distributions = {'cx': ['Uniform', -1.0, 1.0],
                       'cy': ['Uniform', -1.0, 1.0],
                       'l': ['Uniform', 1.0, 2.0], 
                       'alpha': ['Uniform', 0.0, 2*np.pi], 
                       'cz': ['Uniform', 0.1, 0.2],
                       #'density': ['Uniform', 500.0, 1000.0]
                       }
priors = prior(keys, prior_distributions)

# TRUTH
truth = {'cx': 0.25, 'cy': 0.25, 'l': 1.5, 'alpha': np.pi/4, 'cz': 0.1}
data = model(survey_coordinates, truth['cx'], truth['cy'], truth['l'], truth['alpha'], truth['cz'])
np.random.seed(seed=123) # setting the seed just for the noise
noise = np.random.normal(loc=0.0, scale=sigma, size=np.shape(data))
np.random.seed(seed=None)
data = data+noise
survey = GravitySurvey(gravity=data, shape=survey_shape, ranges=ranges)
survey.plot_contours(filename=os.path.join(bilby_outdir, 'survey.png'))

injection_parameters = dict.fromkeys(keys)
for idx, k in enumerate(keys):
     injection_parameters[k] = truth[k]

# LIKELIHOOD
likelihood = bilby.likelihood.GaussianLikelihood(survey_coordinates, data, model, sigma)

# -------------------- Running sampler ---------------------------
result = bilby.run_sampler(
   likelihood=likelihood,
   priors=priors,
   sampler="dynesty",
   nlive = 1000,
   maxmcmc = 10000,
   injection_parameters=injection_parameters,
   outdir=bilby_outdir,
   label=label,
)
result.plot_corner()
