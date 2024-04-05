#!/scratch/wiay/2263373r/masters/conda_envs/venv/bin/python
import os
import pickle as pkl
import torch
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from datetime import datetime

from giflow.flowmodel import FlowModel, save_flow
from giflow.box import BoxDataset

# ------------- Directories ---------------------------------
data_location = '/data/' # THIS needs to be edited to give the data location
save_location = '/results/' # THIS needs to be edited to give the saving location

# ------------- Reading the data ----------------------------
datasize = 500000 # THIS needs to be edited to give the overall desired data set size
survey_coordinates_to_include = [] # THIS needs to be edited if we want to include survey coordinates in the conditional

with open(os.path.join(data_location, f"trainset.pkl"), 'rb') as file:
    dt = pkl.load(file)
train_data, tradin_conditional = dt.make_data_arrays(survey_coordinates_to_include=survey_coordinates_to_include)
train_data = train_data[:datasize,:]
train_conditional = train_conditional[:datasize,:]

with open(os.path.join(data_location, 'validationset.pkl'), 'rb') as file:
   dt = pkl.load(file)
val_data, val_conditional = dt.make_data_arrays(survey_coordinates_to_include=survey_coordinates_to_include)

print(f"Data read. Location: \t {data_location}")

# ------------- Defining scalers ---------------------------
# only fitting to the data to construct scaler, scaling is done within the flow class
sc_data = MinMaxScaler()
sc_data.fit(train_data)
sc_conditional = MinMaxScaler()
sc_conditional.fit(train_conditional.reshape(-1, train_conditional.shape[-1]))
scalers = {'conditional': sc_conditional, 'data': sc_data}

# --------------- Defining the flow ------------------------
device = torch.device('cuda')
# THIS needs to be edited for the hyperparameters of the flow
hyperparameters={'n_inputs': 7,
                 'n_conditional_inputs': 64,
                 'n_transforms': 12,
                 'n_blocks_per_transform': 2,
                 'n_neurons': 64,
                 'batch_norm': True,
                 'batch_size': 5000,
                 'early_stopping': False,
                 'lr': 0.001,
                 'epochs': 2000
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
flow.train(optimiser=optimiser, validation_dataset=val_dataset, train_dataset=train_dataset, scheduler=scheduler, device=device)


