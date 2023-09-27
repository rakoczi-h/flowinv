import numpy as np
import torch
from utils import scale_data, inv_scale_data
import json
from glasflow.flows import RealNVP
from test import *
import h5py
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from plot import *
import os
import imageio

compare_to_bilby = False
bilby_dir = '/data/www.astro/2263373r/bilby_outdir/testcases/'

model_loc = '/data/www.astro/2263373r/grav_inv/run_2023-09-15 17:20:19.895996/'
save_loc = model_loc
with open(model_loc + 'params.json') as json_file:
    params = json.load(json_file)

gpu_num = params['gpu_num']
device = torch.device('cuda')
# Loading the flow
flow = RealNVP(
     n_inputs=params['n_inputs'],
     n_conditional_inputs=params['n_conditional_inputs'],
     n_transforms=params['n_transforms'],
     n_neurons=params['n_neurons'],
     n_blocks_per_transform=params['n_blocks_per_transform'],
     batch_norm_between_transforms=params['batch_norm']
)
flow.load_state_dict(torch.load(model_loc+'flow_model.pt'))
flow.to(device)
flow.eval()
print("Loaded the flow...")

dataloc = params['data_loc']
testname = params['test_name']
dataname = params['data_name']
datasize = params['data_size']
testsize = params['test_size']
sigma = params['survey_noise_scale']

# PARAMETERISED CASE
if params['parameterised']:
    # Reading in the test data
    h = h5py.File(dataloc+testname+'.hdf5', "r")
    test_models_parameterised = np.array(h['models_parameterised'])
    test_surveys = np.array(h['surveys'])
    # Adding a little noise to the survey, only to the gravity not the grid
    if params['add_survey_noise']:
        test_surveys[:,:,0] = test_surveys[:,:,0] + np.random.normal(0, sigma, np.shape(test_surveys[:,:,0]))
    survey_coords = np.array(h['surveys'][:, :, 1:])
    h.close()


    # Reading in the train data
    g = h5py.File(dataloc+dataname+'.hdf5', "r")
    # Not reading in all of it as it is too large
    models_parameterised = np.array(g['models_parameterised'][:10000, :])
    surveys = np.array(g['surveys'][:10000, :, :])
    # adding noise
    if params['add_survey_noise'] == True:
        surveys[:,:,0] = surveys[:,:,0] + np.random.normal(0, sigma, np.shape(surveys[:,:,0]))
    g.close()
    print("Loaded the data...")

    # scaling
    # assumes the scaler.pkl files are available for the above training data set already
    models_parameterised_rescaled = scale_data(models_parameterised, mode='model_parameterised', fit=False, name=dataname, dataloc=dataloc, scaler=params['scaler_model'])
    surveys_toscale = np.reshape(surveys, (np.shape(surveys)[0]*np.shape(surveys)[1], np.shape(surveys)[2]))
    surveys_toscale = scale_data(surveys_toscale, mode='survey', fit=False, name=dataname, dataloc=dataloc, scaler=params['scaler_survey'])
    test_models_parameterised_rescaled= scale_data(test_models_parameterised, mode='model_parameterised', fit=False, name=dataname, dataloc=dataloc, scaler=params['scaler_model'])
    test_surveys_toscale = np.reshape(test_surveys, (np.shape(test_surveys)[0]*np.shape(test_surveys)[1], np.shape(test_surveys)[2]))
    test_surveys_toscale = scale_data(test_surveys_toscale, mode='survey', fit=False, name=dataname, dataloc=dataloc, scaler=params['scaler_survey'])

    surveys_rescaled = np.reshape(surveys_toscale, (np.shape(surveys)[0], np.shape(surveys)[1], np.shape(surveys)[2]))
    test_surveys_rescaled = np.reshape(test_surveys_toscale, (np.shape(test_surveys)[0], np.shape(test_surveys)[1], np.shape(test_surveys)[2]))
    if params['noisygrid']:
        surveys_rescaled = surveys_rescaled[:,:,:3]
        surveys_rescaled = np.reshape(surveys_rescaled, (np.shape(surveys_rescaled)[0], np.shape(surveys_rescaled)[1]*np.shape(surveys_rescaled)[2]))
        test_surveys_rescaled = test_surveys_rescaled[:,:,:3]
        test_surveys_rescaled = np.reshape(test_surveys_rescaled, (np.shape(test_surveys_rescaled)[0], np.shape(test_surveys_rescaled)[1]*np.shape(test_surveys_rescaled)[2]))
    else:
        surveys_rescaled = surveys_rescaled[:,:,0]
        test_surveys_rescaled = test_surveys_rescaled[:,:,0]

    x_train, x_val, y_train, y_val = train_test_split(models_parameterised_rescaled, surveys_rescaled)

    x_train_tensor = torch.from_numpy(x_train.astype(np.float32)).to(device)
    y_train_tensor = torch.from_numpy(y_train.astype(np.float32)).to(device)
    x_val_tensor = torch.from_numpy(x_val.astype(np.float32)).to(device)
    y_val_tensor = torch.from_numpy(y_val.astype(np.float32)).to(device)
    x_test_tensor = torch.from_numpy(test_models_parameterised_rescaled.astype(np.float32)).to(device)
    y_test_tensor = torch.from_numpy(test_surveys_rescaled.astype(np.float32)).to(device)

    print("Testing...")

    start_test = datetime.now()
    # Making p-p plot
    labels = [r'$p_{x}$', r'$p_{y}$', r'$p_{z}$', r'$l_{x}$', r'$l_{y}$', r'$l_{z}$', r'$\alpha_{x}$', r'$\alpha_{y}$']
    keys = ["px", "py", "pz", "lx", "ly", "lz", "alpha_x", "alpha_y"]
    p_p_testing(flow, x_val_tensor, y_val_tensor, n_test_samples=2000, n_test_cases=100, n_params=10, saveloc=model_loc, keys=labels)

    labels_corner = labels
    for i in range(testsize):
        print(f"Test case {i}...")
        # sampling
        num_samples = 10000
        samples, log_probs = sample_and_logprob(num_samples, y_test_tensor[i:i+1,:], flow)
        samples_scaled = inv_scale_data(samples, mode='models_parameterised', name=dataname, dataloc=dataloc, scaler=params['scaler_model'])
        plot_compare_survey(samples_scaled[:10,:], test_surveys[i,:,0], test_surveys[i,:,1:], mode='model_parameterised', contour=True, saveloc=save_loc, filename=f"comp_survey_contour_{i}")
        if compare_to_bilby == True:
            samples_list = []
            with open(bilby_dir + f"results_{i}.json") as json_file:
                bilbyresults = json.load(json_file)
                bilbysamples = np.vstack(bilbyresults['samples']['content'])
                samples_list.append(bilbysamples)
            samples_list.append(samples_scaled)
            truths = test_models_parameterised[i,:]
            overlaid_corner(samples_list, labels_corner, ['dynesty', 'flow'], values=truths, saveloc=save_loc, filename=f"corner_plot_compare_{i}")
        file_path = save_loc+f"testsamples_{i}.hdf5"
        if os.path.isfile(file_path):
            print(f"testsamples_{i}.hdf5 already exists, replacing...")
            os.remove(file_path)
        h = h5py.File(file_path, "a")
        h.create_dataset("samples", data=samples_scaled)
        h.create_dataset("log_probs", data=log_probs)
        h.create_dataset("survey_grid", data=test_surveys[i,:,1:])
        h.create_dataset("survey_true", data=test_surveys[i,:,0])
        h.create_dataset("models_params_true", data=test_models_parameterised[i,:])
        h.close()
    # Latent space testing
    latent_samples, latent_logprobs = forward_and_logprob(x_train_tensor, y_train_tensor, flow)
    mean_kldiv, std_kldiv = KL_divergence_latent(latent_samples)
    plot_latent_hist(latent_samples, mean_kldiv, std_kldiv, save_loc)

# VOXELISED CASE
else:
    h = h5py.File(dataloc+testname+'.hdf5', "r")
    test_models_parameterised = np.array(h['models_parameterised'])
    test_models = np.array(h['models'])
    voxel_grid = np.array(h['voxel_grid'])
    voxel_grid_plot = np.zeros(np.shape(voxel_grid))
    voxel_grid_plot[:,0,:] = voxel_grid[:,0,:]
    voxel_grid_plot[:,1,:] = voxel_grid[:,1,:]
    voxel_grid_plot[:,2,:] = voxel_grid[:,2,:] - 60
    test_surveys = np.array(h['surveys'])
    if params['add_survey_noise']:
        test_surveys[:,:,0] = test_surveys[:,:,0] + np.random.normal(0, sigma, np.shape(test_surveys[:,:,0]))
    survey_coords = np.array(h['surveys'][:, :, 1:])
    h.close()

    g = h5py.File(dataloc+dataname+'.hdf5', "r")
    surveys = np.array(g['surveys'][:10000, :, :])
    if add_survey_noise == True:
        surveys[:,:,0] = surveys[:,:,0] + np.random.normal(0, sigma, np.shape(surveys[:,:,0]))
    models_parameterised = np.array(g['models_parameterised'][:10000,:])
    models = np.array(g['models'][:10000, :])
    g.close()
    print("Loaded the test data...")

    models_rescaled = scale_data(models, mode='model', fit=False, name=dataname, dataloc=dataloc, scaler=params['scaler_model'])
    surveys_toscale = np.reshape(surveys, (np.shape(surveys)[0]*np.shape(surveys)[1], np.shape(surveys)[2]))
    surveys_toscale = scale_data(surveys_toscale, mode='survey', fit=False, name=dataname, dataloc=dataloc, scaler=params['scaler_survey'])
    test_models_rescaled= scale_data(test_models, mode='model', fit=False, name=dataname, dataloc=dataloc, scaler=params['scaler_model'])
    test_surveys_toscale = np.reshape(test_surveys, (np.shape(test_surveys)[0]*np.shape(test_surveys)[1], np.shape(test_surveys)[2]))
    test_surveys_toscale = scale_data(test_surveys_toscale, mode='survey', fit=False, name=dataname, dataloc=dataloc, scaler=params['scaler_survey'])

    surveys_rescaled = np.reshape(surveys_toscale, (np.shape(surveys)[0], np.shape(surveys)[1], np.shape(surveys)[2]))
    test_surveys_rescaled = np.reshape(test_surveys_toscale, (np.shape(test_surveys)[0], np.shape(test_surveys)[1], np.shape(test_surveys)[2]))
    if params['noisy_grid']:
        surveys_rescaled = surveys_rescaled[:,:,:3]
        surveys_rescaled = np.reshape(surveys_rescaled, (np.shape(surveys_rescaled)[0], np.shape(surveys_rescaled)[1]*np.shape(surveys_rescaled)[2]))
        test_surveys_rescaled = test_surveys_rescaled[:,:,:3]
        test_surveys_rescaled = np.reshape(test_surveys_rescaled, (np.shape(test_surveys_rescaled)[0], np.shape(test_surveys_rescaled)[1]*np.shape(test_surveys_rescaled)[2]))
    else:
        surveys_rescaled = surveys_rescaled[:,:,0]
        test_surveys_rescaled = test_surveys_rescaled[:,:,0]

    x_train, x_val, y_train, y_val = train_test_split(models_rescaled, surveys_rescaled)

    x_train_tensor = torch.from_numpy(x_train.astype(np.float32)).to(device)
    y_train_tensor = torch.from_numpy(y_train.astype(np.float32)).to(device)
    x_val_tensor = torch.from_numpy(x_val.astype(np.float32)).to(device)
    y_val_tensor = torch.from_numpy(y_val.astype(np.float32)).to(device)
    x_test_tensor = torch.from_numpy(test_models_rescaled.astype(np.float32)).to(device)
    y_test_tensor = torch.from_numpy(test_surveys_rescaled.astype(np.float32)).to(device)

    print("Testing...")
    start_test = datetime.now()
    # Making p-p plot
    labels = ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10"]
    p_p_testing(flow, x_val_tensor, y_val_tensor, n_test_samples=1000, n_params=10, saveloc=model_loc, keys=labels)

    for i in range(testsize):
        print(f"Test case {i}...")
        num_samples = 10000
        samples, log_probs = sample_and_logprob(num_samples, y_test_tensor[i:i+1,:], flow)
        samples_scaled = inv_scale_data(samples, mode='model', name=dataname, dataloc=dataloc, scaler='minmax')
        plot_compare_survey(samples_scaled[:10,:], test_surveys[i,:,0], test_surveys[i,:,1:], mode='model', saveloc=save_loc, contour=True, filename=f"comp_survey_contour_{i}", voxel_grid=voxel_grid)
        plot_compare_voxel_slices(samples_scaled, log_probs, test_models[i,:], saveloc=save_loc, filename=f"compare_voxel_slices_{i}")
        file_path = save_loc+f"testsamples_{i}.hdf5"
        if os.path.isfile(file_path):
            print(f"testsamples_{i}.hdf5 already exists, replacing...")
            os.remove(file_path)
        h = h5py.File(file_path, "a")
        h.create_dataset("samples", data=samples_scaled)
        h.create_dataset("log_probs", data=log_probs)
        h.create_dataset("survey_grid", data=test_surveys[i,:,1:])
        h.create_dataset("survey_true", data=test_surveys[i,:,0])
        h.create_dataset("models_true", data=test_models[i,:])
        h.create_dataset("voxel_grid", data=voxel_grid)
        h.close()

    # Latent space testing
    latent_samples, latent_logprobs = forward_and_logprob(x_train_tensor, y_train_tensor, flow)
    mean_kldiv, std_kldiv = KL_divergence_latent(latent_samples)
    plot_latent_hist(latent_samples, mean_kldiv, std_kldiv, save_loc)

