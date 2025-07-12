import bilby
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl

from giflow.fault import Fault
from giflow.survey import GravitySurvey


label = "6_parameters_dl"
bilby_outdir = '/data/www.astro/2263373r/fault_python_version/bilby/'
bilby.utils.check_directory_exists_and_if_not_mkdir(bilby_outdir)
bilby_outdir = os.path.join(bilby_outdir, label)
bilby.utils.check_directory_exists_and_if_not_mkdir(bilby_outdir)

# ----------------------- Functions -------------------------------
def model(survey_coordinates, cx, cy, l, alpha, cz, DL_ratio):
    """
    Function defining the forward model.
    """
    fault = Fault(parameters = {"cx": cx, "cy": cy, "l": l, "alpha": alpha, "cz": cz,
                                 "density": 800.0,
                                 "DL_ratio": DL_ratio,
                                 "dip": 70*np.pi/180})
    X = np.linspace(-2, 2, num=50)
    Y = np.linspace(-2, 2, num=50)
    X, Y = np.meshgrid(X, Y)
    X = np.expand_dims(X, axis=2)
    Y = np.expand_dims(Y, axis=2)
    Z = np.zeros(np.shape(X))
    grid = np.c_[X, Y, Z]

    fault.make_fault(grid=grid)
    gz, _ = fault.forward_model(survey_coordinates = survey_coordinates.copy())

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

# ---------------------- Reading the data ----------------------
data_location = '/scratch/balta1/2263373r/fault/gzBA_eFTG_Survey_2_padded.pkl'
with open(data_location, 'rb') as file:
    real = pkl.load(file)
extent = (real.ranges[0][0], real.ranges[0][1], real.ranges[1][1], real.ranges[1][0])
plt.imshow(np.rot90(real.gravity.reshape((50,50))), extent=extent)
plt.colorbar()
plt.savefig(os.path.join(bilby_outdir, 'survey.png'))
plt.close()

survey_coordinates = real.survey_coordinates

# --------------------- Defining sampler inputs ------------------
keys = ['cx', 'cy', 'l', 'alpha', 'cz', 'DL_ratio']
sigma = 0.1

# SURVEY GRID
survey_shape = [50,50]
X = np.linspace(-2, 2, num=survey_shape[0])
Y = np.linspace(-2, 2, num=survey_shape[1])
X, Y = np.meshgrid(X, Y)
X = np.expand_dims(X, axis=2)
Y = np.expand_dims(Y, axis=2)
Z = np.zeros(np.shape(X))
survey_coordinates = np.c_[X.flatten(), Y.flatten(), Z.flatten()]

# PRIOR
prior_distributions = {'cx': ['Uniform', -1.0, 1.0], 'cy': ['Uniform', -1.0, 1.0], 'l': ['Uniform', 1.0, 2.0], 'alpha': ['Uniform', 0.0, 2*np.pi], 'cz': ['Uniform', 0.1, 0.2], 'DL_ratio': ['Uniform', 0.001, 1.0]}
priors = prior(keys, prior_distributions)

# TRUTH
truth = {'cx': 0.25, 'cy': 0.5, 'l': 1.5, 'alpha': np.pi/4, 'cz': 0.1, 'DL_ratio': 0.02}
data = model(survey_coordinates, truth['cx'], truth['cy'], truth['l'], truth['alpha'], truth['cz'], truth['DL_ratio'])
noise = np.random.normal(loc=0.0, scale=sigma, size=np.shape(data))
data = data+noise
survey = GravitySurvey(gravity=data, shape=survey_shape, survey_coordinates = survey_coordinates)
survey.plot_pixels(filename=os.path.join(bilby_outdir, 'survey.png'))

injection_parameters = dict.fromkeys(keys)
for idx, k in enumerate(keys):
     injection_parameters[k] = truth[k]

# LIKELIHOOD
likelihood = bilby.likelihood.GaussianLikelihood(survey_coordinates, data, model, sigma)

# # -------------------- Running sampler ---------------------------
result = bilby.run_sampler(
   likelihood=likelihood,
   priors=priors,
   sampler="dynesty",
   nlive=1000,
   maxmcmc = 10000,
   injection_parameters=injection_parameters,
   outdir=bilby_outdir,
   label=label,
)
result.plot_corner()
