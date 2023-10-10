from datetime import datetime
import json
from glasflow.flows import RealNVP
import numpy as np
import os
import sys
import h5py
from sklearn.model_selection import train_test_split
import torch
import joblib

from lib.utils import scale_data
from lib.plot import plot_flow_diagnostics, plot_loss
from lib.train import train
from lib.test import forward_and_logprob, KL_divergence_latent
np.random.seed(1434)

# ---------------SETTING UP--------------------
start_time = datetime.now()
print(f"Start of new run: {start_time}")
with open('flow_params.json') as json_file:
    params = json.load(json_file)

datafile = sys.argv[1] # Passing the data as an input file
savedir = sys.argv[2] # Passing the output directory as an input
print("----------------------------------------")
print(f"Input file: {datafile}")
print(f"Output directory: {savedir}")
if not os.path.exists(savedir):
    print("Output directory already exists, contents will be overwritten.")
if not os.path.exists(savedir):
    os.mkdir(savedir)
print("----------------------------------------")

# Device:
gpu_num = params['gpu_num']
device = torch.device("cuda:%d" % gpu_num if torch.cuda.is_available() else "cpu")

# ------------------MAKING DATA------------------------
datasize = params['data_size'] # This is the size of the data set that is used for the training.
f = h5py.File(datafile, "r")
# Reading in survey data
surveys = np.array(f['surveys'][:datasize,:,:])
if params['add_survey_noise']:
    sigma = params['survey_noise_scale']
surveys[:,:,0] = surveys[:,:,0] + np.random.normal(0, sigma, np.shape(surveys[:,:,0]))
surveys_toscale = np.reshape(surveys, (np.shape(surveys)[0]*np.shape(surveys)[1], np.shape(surveys)[2]))
surveys_toscale = scale_data(surveys_toscale, mode='survey', fit=True, name=dataname, dataloc=dataloc, scaler=params['scaler_survey'])
surveys = np.reshape(surveys_toscale, np.shape(surveys))
if params['noisygrid']:
    surveys = surveys[:,:,:3]
    surveys = np.reshape(surveys, (np.shape(surveys)[0], np.shape(surveys)[1]*np.shape(surveys)[2]))
else:
    surveys = surveys[:,:,0]

# Reading in the source models
if params['parameterised']:
    models = np.array(f['models_parameterised'][:datasize,:])
    models = scale_data(models, mode='model_parameterised', fit=True, name=dataname, dataloc=dataloc, scaler=params['scaler_model'])
else:
    models = np.array(f['models'][:datasize,:])
    models = scale_data(models, mode='model', fit=True, name=dataname, dataloc=dataloc, scaler=params['scaler_model'])
f.close()

print("Loaded the dataset...")
print("----------------------------------------")
print(f"Model data size: \t {np.shape(models)}")
print(f"Survey data  size: \t {np.shape(surveys)}")
print("----------------------------------------")

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
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Saving parmeters file and making save directory
os.mkdir(savedir + 'run_' + str(start_time))
saveloc = savedir + 'run_' + str(start_time) + '/'
out_file = open(os.path.join(saveloc, "params.json"), "w")
json.dump(params, out_file, indent=4)
out_file.close()

# --------------------------------------Network----------------------------------------------
# Creating the flow
flow = RealNVP(
    n_inputs=params['n_inputs'],
    n_transforms=params['n_transforms'],
    n_conditional_inputs=params['n_conditional_inputs'],
    n_neurons=params['n_neurons'],
    n_blocks_per_transform=params['n_blocks_per_transform'],
    batch_norm_between_transforms=params['batch_norm'], #!
    # linear_transform=params['linear_transform']
)
flow.to(device)
# Here the optimiser can be edited
optimiser = torch.optim.Adam(flow.parameters(), lr=params['learning_rate'])
# Here the scheduler can be edited
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.05, patience=100, cooldown=10,
                                                       min_lr=1e-6, verbose=True)
print(f"Created flow and sent to {device}...")
print(f"Network parameters:")
print("----------------------------------------")
print(f"n_inputs: \t\t {params['n_inputs']}")
print(f"n_conditional_inputs: \t {params['n_conditional_inputs']}")
print(f"n_transforms: \t\t {params['n_transforms']}")
print(f"n_blocks_per_trans: \t {params['n_blocks_per_transform']}")
print(f"n_neurons: \t\t {params['n_neurons']}")
print(f"batch_norm: \t\t {params['batch_norm']}")
print(f"optimiser: \t\t {type (optimiser).__name__}")
print(f"scheduler: \t\t {type (scheduler).__name__}")
print(f"early stopping: \t {params['early_stopping']}")
print(f"initial learning rate: \t {params['learning_rate']}")
print("----------------------------------------")
# Training
epochs = params['epochs']
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
        latent_samples, latent_logprobs = forward_and_logprob(x_train_tensor[:10000,:], y_train_tensor[:10000, :], flow)
        mean_kldiv, std_kldiv = KL_divergence_latent(latent_samples)
        plot_flow_diagnostics(latent_samples, latent_logprobs, loss, mean_kldiv, saveloc=saveloc, filename='diagnostics.png')
        end_test = datetime.now()
        print(f"Finished testing, time taken: \t {end_test-start_test}")
    # Setting early stopping condition
    if loss['val'][-1] < min_val_loss:
        min_val_loss = loss['val'][-1]
        iters_no_improve = 0
        best_model = flow.state_dict()
    else:
        iters_no_improve += 1
    if (params['early_stopping'] and iters_no_improve == params['patience']):
        torch.save(best_model, os.path.join(saveloc, 'flow_model.pt'))
        print("Early stopping!")
        break
    end_epoch = datetime.now()
    if not i % 10:
        torch.save(best_model, os.path.join(saveloc, 'flow_model.pt'))
        print(f"Epoch {i} \t train: {loss['train'][-1]:.3f} \t val: {loss['val'][-1]:.3f} \t t: {end_epoch-start_epoch}")
flow.eval()
print('Finished training...')

# Plotting the loss
plot_loss(loss, saveloc=saveloc)

end_time = datetime.now()
print("Run time: ", end_time-start_time)

