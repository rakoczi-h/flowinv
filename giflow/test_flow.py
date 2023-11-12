import h5py
import sys
import os
import numpy as np
import json
import torch
from glasflow.flows import RealNVP
from datetime import datetime

from lib.plot import plot_compare_survey, corner_plot
from lib.test import p_p_testing, sample_and_logprob
from lib.utils import scale_data, inv_scale_data

datadir = sys.argv[1] # Folder containing the data to be used
flowloc = sys.argv[2] # Path to the directory which contains the flow model and corresponding parameter file

print("----------------------------------------")
print(f"Input data directory: {datadir}")
print(f"Flow model directory: {flowloc}")
print("----------------------------------------")

with open(os.path.join(flowloc, 'params.json')) as json_file: # Reading the parameters file
    params = json.load(json_file)

# Device
gpu_num = params['gpu_num']
device = torch.device("cuda:%d" % gpu_num if torch.cuda.is_available() else "cpu")

savedir = os.path.join(flowloc, 'plots/') # This is where the plots will be saved
if not os.path.exists(savedir):
    os.mkdir(savedir)

# --------------------------- TEST FILE ---------------------------
testfile = h5py.File(os.path.join(datadir, 'testdata.hdf5'), "r")
testsize = params['test_size']
# Reading in the survey data
test_surveys = np.array(testfile['surveys'])
if params['add_survey_noise']:
    sigma = params['survey_noise_scale']
    test_surveys[:,:,0] = test_surveys[:,:,0] + np.random.normal(0, sigma, np.shape(test_surveys[:,:,0]))
surveys_toscale = np.reshape(test_surveys, (np.shape(test_surveys)[0]*np.shape(test_surveys)[1], np.shape(test_surveys)[2]))
surveys_toscale = scale_data(surveys_toscale, mode='survey', fit=False, name='', scalerloc=flowloc, scaler=params['scaler_survey'])
test_surveys_scaled = np.reshape(surveys_toscale, np.shape(test_surveys))
if params['noisygrid']:
    test_surveys = test_surveys[:,:,:3]
    test_surveys_scaled = test_surveys_scaled[:,:,:3]
    test_surveys = np.reshape(test_surveys, (np.shape(test_surveys)[0], np.shape(test_surveys)[1]*np.shape(test_surveys)[2]))
    test_surveys_scaled = np.reshape(test_surveys_scaled, (np.shape(test_surveys_scaled)[0], np.shape(test_surveys_scaled)[1]*np.shape(test_surveys_scaled)[2]))
else:
    test_surveys = test_surveys[:,:,0]
    test_surveys_scaled = test_surveys_scaled[:,:,0]

# Reading in the source models
if params['parameterised']:
    test_models = np.array(testfile['models_parameterised'])
    test_models_scaled = scale_data(test_models, mode='model_parameterised', fit=False, name='', scalerloc=flowloc, scaler=params['scaler_model'])
else:
    test_models = np.array(testfile['models'])
    test_models_scaled = scale_data(test_models, mode='model', fit=True, name='', scalerloc=flowloc, scaler=params['scaler_model'])
testfile.close()

x_test_tensor = torch.from_numpy(test_models_scaled.astype(np.float32)).to(device)
y_test_tensor = torch.from_numpy(test_surveys_scaled.astype(np.float32)).to(device)
#------------------------ TRAINING DATA FILE ---------------------------
datafile = h5py.File(os.path.join(datadir, 'traindata.hdf5'), "r")
datasize = 10000 # NOT READING IN THE WHOLE DATASET
# Reading in survey data
surveys = np.array(datafile['surveys'][:datasize,:,:])
if params['add_survey_noise']:
    sigma = params['survey_noise_scale']
    surveys[:,:,0] = surveys[:,:,0] + np.random.normal(0, sigma, np.shape(surveys[:,:,0]))
surveys_toscale = np.reshape(surveys, (np.shape(surveys)[0]*np.shape(surveys)[1], np.shape(surveys)[2]))
surveys_toscale = scale_data(surveys_toscale, mode='survey', fit=True, name='', scaleraloc=flowloc, scaler=params['scaler_survey'])
surveys_scaled = np.reshape(surveys_toscale, np.shape(surveys))
if params['noisygrid']:
    surveys = surveys[:,:,:3]
    surveys_scaled = surveys_scaled[:,:,:3]
    surveys_scaled = np.reshape(surveys_scaled, (np.shape(surveys_scaled)[0], np.shape(surveys_scaled)[1]*np.shape(surveys_scaled)[2]))
    surveys = np.reshape(surveys, (np.shape(surveys)[0], np.shape(surveys)[1]*np.shape(surveys)[2]))
else:
    surveys = surveys[:,:,0]
    surveys_scaled = surveys_scaled[:,:,0]

# Reading in the source models
if params['parameterised']:
    models = np.array(datafile['models_parameterised'][:datasize,:])
    models_scaled = scale_data(models, mode='model_parameterised', fit=False, name='', scalerloc=flowloc, scaler=params['scaler_model'])
else:
    models = np.array(datafile['models'][:datasize,:])
    models_scaled = scale_data(models, mode='model', fit=False, name='', scalerloc=flowloc, scaler=params['scaler_model'])
datafile.close()

print("Loaded the data...")

x_train_tensor = torch.from_numpy(models_scaled.astype(np.float32)).to(device)
y_train_tensor = torch.from_numpy(surveys_scaled.astype(np.float32)).to(device)
# --------------------------- The Flow ---------------------------
gpu_num = params['gpu_num']
device = torch.device("cuda:%d" % gpu_num if torch.cuda.is_available() else "cpu")
flow = RealNVP(
    n_inputs=params['n_inputs'],
    n_transforms=params['n_transforms'],
    n_conditional_inputs=params['n_conditional_inputs'],
    n_neurons=params['n_neurons'],
    n_blocks_per_transform=params['n_blocks_per_transform'],
    batch_norm_between_transforms=params['batch_norm']
)
flow.load_state_dict(torch.load(os.path.join(flowloc, 'flow_model.pt')))
flow.to(device)
flow.eval()
print(f"Loaded the flow and sent to {device}...")

# --------------------------- P-P  Plot ---------------------------
labels = [r'$p_x$', r'$p_y$', r'$p_z$', r'$l_x$', r'$l_y$', r'$l_z$', r'$alpha_x$', r'$alpha_y$']
p_p_testing(flow, x_train_tensor, y_train_tensor, n_test_samples=1000, n_params=len(labels), saveloc=savedir, keys=labels)

# -------------------------- Test Cases ---------------------------
for i in range(testsize):
    print(f"Test case {i}...")
    testdir = os.path.join(savedir, f"testcase_{i}")
    if not os.path.exists(testdir):
        os.mkdir(testdir)
    # Sampling
    num_samples = 10000
    start_sample = datetime.now()
    samples, log_probs = sample_and_logprob(num_samples, y_test_tensor[i:i+1,:], flow)
    end_sample = datetime.now()
    print(f"Time taken with sampling {end_sample-start_sample}")
    # Scaling the samples
    samples_scaled = inv_scale_data(samples, mode='model_parameterised', name='', scalerloc=flowloc, scaler=params['scaler_model'])
    # Making survey comparison plot
    plot_compare_survey(samples_scaled[:10,:], test_surveys[i,:,0], test_surveys[i,:,1:], mode='model_parameterised', contour=True, saveloc=testdir, filename=f"comp_survey_contour_{i}")
    # Making corner plot
    corner_plot(samples_scaled, labels, values=test_models[i,:], saveloc=testdir, filename=f"corner_plot_{i}.png")

print("Finished testing...")
