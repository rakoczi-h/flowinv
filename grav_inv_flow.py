#!/scratch/wiay/2263373r/masters/conda_envs/venv/bin/python

import scipy
from datetime import datetime
import json
from glasflow.flows import RealNVP
import numpy as np
import matplotlib
import os
import h5py
from sklearn.model_selection import train_test_split
from plot import *
from utils import scale_data
import torch
import matplotlib.pyplot as plt
import joblib
import warnings
from flow_functions import *
from train import train

# ------------------- OPTIONS --------------------
noisygrid = False
parameterised = True
add_survey_noise = True

np.random.seed(1434)
# ---------------SETTING UP--------------------
start_time = datetime.now()
print(f"Start of new run: {start_time}")
with open('params.json') as json_file:
    params = json.load(json_file)
# Directories:
savedir = params['save_dir']
dataloc = params['data_loc']
datasize = params['data_size']
testname = params['test_name']
os.mkdir(savedir + 'run_' + str(start_time))
saveloc = savedir + 'run_' + str(start_time) + '/'
# Device:
gpu_num = params['gpu_num']
# device = torch.device("cuda:%d" % gpu_num if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
out_file = open(saveloc+"params.json", "w")

json.dump(params, out_file, indent=4)
out_file.close()
print("Set up script...")

# ------------------MAKING DATA------------------------
sigma = params['survey_noise_scale']

dataname = params['data_name']
f = h5py.File(dataloc+dataname+'.hdf5', "r")
surveys = np.array(f['surveys'][:datasize,:,:])
surveys[:,:,0] = surveys[:,:,0] + np.random.normal(0, sigma, np.shape(surveys[:,:,0]))
surveys_toscale = np.reshape(surveys, (np.shape(surveys)[0]*np.shape(surveys)[1], np.shape(surveys)[2]))
if parameterised:
    models = np.array(f['models_parameterised'][:datasize,:])
    models = scale_data(models, mode='model_parameterised', fit=True, name=dataname, dataloc=dataloc, scaler=params['scaler_model'])
else:
    models = np.array(f['models'][:datasize,:])
    models = scale_data(models, mode='model', fit=True, name=dataname, dataloc=dataloc, scaler=params['scaler_model'])

print("Loaded the dataset...")
print("Model data size: ", np.shape(models))
print("Survey data  size: ", np.shape(surveys))

# scaling the survey
surveys_toscale = scale_data(surveys_toscale, mode='survey', fit=True, name=dataname, dataloc=dataloc, scaler=params['scaler_survey'])

surveys = np.reshape(surveys_toscale, np.shape(surveys))
if noisygrid:
    surveys = surveys[:,:,:3]
    surveys = np.reshape(surveys, (np.shape(surveys)[0], np.shape(surveys)[1]*np.shape(surveys)[2]))
else:
    surveys = surveys[:,:,0]

# Formatting data
x_train, x_val, y_train, y_val = train_test_split(models, surveys)

batch_size = params['batch_size']
x_train_tensor = torch.from_numpy(x_train.astype(np.float32)).to(device)
y_train_tensor = torch.from_numpy(y_train.astype(np.float32)).to(device)
train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

x_val_tensor = torch.from_numpy(x_val.astype(np.float32)).to(device)
y_val_tensor = torch.from_numpy(y_val.astype(np.float32)).to(device)
val_dataset = torch.utils.data.TensorDataset(x_val_tensor, y_val_tensor)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# ------------------------Network----------------------------------------------
# Creating the flow
flow = RealNVP(
    n_inputs=params['n_inputs'],
    n_transforms=params['n_transforms'],
    n_conditional_inputs=params['n_conditional_inputs'],
    n_neurons=params['n_neurons'],
    n_blocks_per_transform=params['n_blocks_per_transform'], # this is equivalent to the number of layers
    batch_norm_between_transforms=params['batch_norm'], #!
    # linear_transform=params['linear_transform']
)
flow.to(device)
optimiser = torch.optim.Adam(flow.parameters(), lr=params['learning_rate'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.05, patience=100, cooldown=10,
                                                       min_lr=1e-6, verbose=True)
print(f'Created flow and sent to {device}...')
# Training
epochs = params['epochs']
early_stopping = params['early_stopping']
iters_no_improve = 0
min_val_loss = np.inf
loss = dict(train=[], val=[])
for i in range(epochs):
    start_epoch = datetime.now()
    train_loss, val_loss = train(flow, optimiser, val_loader, train_loader, device)
    loss['train'].append(train_loss)
    loss['val'].append(val_loss)
    scheduler.step(loss['val'][-1])
    # Plotting the loss
    if not i % params['loss_plot_freq']:
        plot_loss(loss, saveloc=saveloc)
    # Testing
    if not i % params['test_freq'] and i != 0:
        print("Testing...")
        start_test = datetime.now()
        flow.eval()
        latent_samples, latent_logprobs = forw_and_logprob(x_train_tensor[:10000,:], y_train_tensor[:10000, :], flow)
        mean_kldiv, std_kldiv = latent_kldiv(latent_samples)
        mean_kldiv_ref, std_kldiv_ref = latent_kldiv(np.random.normal(loc=0.0, scale=1.0, size=np.shape(latent_samples)))
        plot_latent_corr(latent_samples, saveloc)
        plot_latent_logprob(latent_logprobs, saveloc)
        plot_latent_hist(latent_samples, mean_kldiv, std_kldiv, mean_kldiv_ref=mean_kldiv_ref, std_kldiv_ref=std_kldiv_ref, saveloc=saveloc)
        end_test = datetime.now()
        print(f"Finished testing, time taken: \t {end_test-start_test}")
    # Setting early stopping condition
    if loss['val'][-1] < min_val_loss:
        min_val_loss = loss['val'][-1]
        iters_no_improve = 0
        best_model = flow.state_dict()
    else:
        iters_no_improve += 1
    if (early_stopping and iters_no_improve == params['patience']):
        torch.save(best_model, saveloc+'flow_model.pt')
        print("Early stopping!")
        break
    end_epoch = datetime.now()
    if not i % 10:
        torch.save(best_model, saveloc+'flow_model.pt')
        print(f"Epoch {i} \t train: {loss['train'][-1]:.3f} \t val: {loss['val'][-1]:.3f} \t t: {end_epoch-start_epoch}")
flow.eval()
print('Finished training...')

# Plotting the loss
plot_loss(loss, saveloc=saveloc)

end_time = datetime.now()
print("Run time: ", end_time-start_time)

