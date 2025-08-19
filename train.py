#!/scratch/balta0/2263373r/conda_envs/giflow/bin/python
import os
import pickle as pkl
import torch
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from datetime import datetime
import sys

from giflow.scaler import Scaler
from giflow.read_files import read_files
from giflow.flowmodel import FlowModel, save_flow
from giflow.box import BoxDataset

# ------------- Directories ---------------------------------
data_location = '/scratch/balta1/2263373r/4_paper/parameterised/'  # THIS needs to be edited to give the data location
save_dir = '/data/www.astro/2263373r/giflow/4_paper/parameterised/' # THIS needs to be edited to give the saving location
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


survey_coordinates_to_include = [] # THIS needs to be edited if we want to include survey coordinates in the conditional
model_info_to_include=[]
mix_survey_order = False

# ------------- Defining scalers ---------------------------
datasize = 500000 # THIS needs to be edited to give the overall desired data set size
train_data, train_conditional = read_files(data_location=data_location, filenames=[f"trainset_{i}.pkl" for i in range(1)], datasize=datasize, survey_coordinates_to_include=survey_coordinates_to_include, model_info_to_include=model_info_to_include, mix_survey_order=mix_survey_order, noise_augment_factor=1)

print(np.shape(train_data))

scalers = [MinMaxScaler()]
sc_data = Scaler(scalers=scalers)
sc_data.scale_data(train_data, fit=True)

scalers = [MinMaxScaler()]
#scalers = [MinMaxScaler()]
sc_conditional=Scaler(scalers=scalers)
sc_conditional.scale_data(train_conditional, fit=True)

scalers = {'conditional': sc_conditional, 'data': sc_data}

# ------------- Reading the data ---------------------------
datasize = 1000000 # THIS needs to be edited to give the overall desired data set size
train_data, train_conditional = read_files(data_location=data_location, filenames=[f"trainset_{i}.pkl" for i in range(2)], datasize=datasize, survey_coordinates_to_include=survey_coordinates_to_include, model_info_to_include=model_info_to_include, mix_survey_order=mix_survey_order, noise_augment_factor=1)
datasize = np.shape(train_data[0])[0]

print(np.shape(train_data))
valsize = 100000 # THIS needs to be edited to give the overall desired data set size
val_data, val_conditional = read_files(data_location=data_location, filenames=[f"validationset_{i}.pkl" for i in range(1)], datasize=valsize, survey_coordinates_to_include=survey_coordinates_to_include, model_info_to_include=model_info_to_include, mix_survey_order=mix_survey_order, noise_augment_factor=1)
valsize = np.shape(val_data[0][0])[0]

print(np.shape(val_data))
print(f"Data read. Location: \t {data_location}")

# ------------- Defining the prior ---------------------
with open(os.path.join(data_location, "validationset_0.pkl"), 'rb') as file:
    dt_val = pkl.load(file)
priors = dt_val.priors
# --------------- Defining the flow ------------------------
start_time = datetime.now()
save_location = os.path.join(save_dir, 'run_'+str(start_time))
os.mkdir(save_location)

device = torch.device('cuda')
# THIS needs to be edited for the hyperparameters of the flow
hyperparameters={'n_inputs': 7,
                 'n_conditional_inputs': 64,
                 'n_transforms': 12,
                 'n_blocks_per_transform': 2,
                 'n_neurons': 64,
                 'batch_norm': True,
                 'batch_size': 5000,
                 'early_stopping': True,
                 'lr': 0.001,
                 'epochs': 300
}
flow = FlowModel(hyperparameters=hyperparameters, datasize=datasize, scalers=scalers)
flow.save_location = save_location
flow.data_location = data_location
save_flow(flow)
flow.construct()

# Defining the flow inputs and training scheduler/optimiser
optimiser = torch.optim.Adam(flow.flowmodel.parameters(), lr=flow.hyperparameters['lr'])
# scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=500, gamma=0.5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.05, patience=90, cooldown=10,
                                                       min_lr=1e-6, verbose=True)

train_dataset = flow.make_tensor_dataset(train_data, train_conditional, device=device, scale=True)
val_dataset = flow.make_tensor_dataset(val_data, val_conditional, device=device, scale=True)

# ----------------- Training the flow ----------------------
flow.train(optimiser=optimiser, validation_dataset=val_dataset, train_dataset=train_dataset, scheduler=scheduler, device=device, prior=priors)


