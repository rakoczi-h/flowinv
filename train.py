import os
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
import torch
import pickle as pkl
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from giflow.box import BoxDataset
from giflow.scaler import Scaler
from giflow.flowmodel import FlowModel, save_flow
from giflow.datareader import DataReader
from giflow.prior import Prior

# Defining directories
data = '/scratch/balta0/2263373r/fault_python/real_inversion/' # where our training and validation that are located
save = '/data/www.astro/2263373r/fault_python_version/real_inversion/' # where we want to save our outputs
save = os.path.join(save, f"run_{datetime.now()}/")

if not os.path.exists(save):
    os.mkdir(save)

# ------------------- DATA ------------------------
model_info_to_include = ['cx', 'cy', 'l', 'alpha', 'cz', 'DL_ratio', 'density']
survey_info_to_include = ['survey_width_ratio', 'noise_scale']
noise_distribution = Prior(distributions={'noise_scale': ['LogUniform', 1e-5, 1.0]})


# Reading in files
trainsize = 500000
dr_train = DataReader(filenames=[f"trainset_{n}.pkl" for n in range(0,50)], data_location=data, model_info_to_include=model_info_to_include, survey_info_to_include=survey_info_to_include, datasize=trainsize)
train_data, train_conditional = dr_train.read_files(noise_distribution=noise_distribution)


if noise_distribution.distributions['noise_scale'][0] == 'LogUniform':
    train_conditional[2] = np.log(train_conditional[2])

# Scaling the data
sc_data = Scaler(scalers = [MinMaxScaler(), MinMaxScaler(), MinMaxScaler(), MinMaxScaler(), MinMaxScaler(), MinMaxScaler(), MinMaxScaler()]) # Need to define the scaler for each element in the train_data list.
sc_data.scale_data(train_data, fit = True) # Fit the scaler and store in the class

sc_conditional = Scaler(scalers = [MinMaxScaler(), MinMaxScaler(), MinMaxScaler()])
sc_conditional.scale_data(train_conditional, fit = True)

scalers = {'conditional': sc_conditional, 'data': sc_data}

trainsize = 1000000
dr_train = DataReader(filenames=[f"trainset_{n}.pkl" for n in range(0,100)], data_location=data, model_info_to_include=model_info_to_include, survey_info_to_include=survey_info_to_include, datasize=trainsize)
train_data, train_conditional = dr_train.read_files(noise_distribution=noise_distribution)

if noise_distribution.distributions['noise_scale'][0] == 'LogUniform':
    train_conditional[2] = np.log(train_conditional[2])

valsize = 100000
dr_train = DataReader(filenames=[f"validationset_{n}.pkl" for n in range(0,100)], data_location=data, model_info_to_include=model_info_to_include, survey_info_to_include=survey_info_to_include, datasize=valsize)
validation_data, validation_conditional = dr_train.read_files(noise_distribution=noise_distribution)

if noise_distribution.distributions['noise_scale'][0] == 'LogUniform':
    validation_conditional[2] = np.log(validation_conditional[2])

# ------------------ FLOW --------------------------
# Defining the flow parameters
hyperparameters = {
        'n_inputs': 7, # the total number of parameters in the source model, including any additional information we chose to include
        'n_conditional_inputs': 2502, # the total number of values in the conditional
        'n_transforms': 8,
        'n_blocks_per_transform': 2,
        'n_neurons': 8,
        # The parameters below define some settings for the training
        'batch_size': 5000,
        'batch_norm': True,
        'lr': 0.001,
        'epochs': 3000,
        'early_stopping': True # if set True, the training stops when the validation loss stops decreasing
}

# Construct the flow
flow = FlowModel(
        hyperparameters = hyperparameters,
        datasize = trainsize,
        scalers = scalers
)
flow.save_location = save
flow.data_location = data
save_flow(flow)
flow.construct()

# Defining the optimiser
optimiser = torch.optim.Adam(
    flow.flowmodel.parameters(),
    lr = flow.hyperparameters['lr']
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimiser, factor=0.1, patience=100, threshold=1e-6, cooldown=10)

# Specifying the GPU
device = torch.device('cuda')


# ------------------- TRAIN ------------------------
# Make the tensor data sets
train_dataset = flow.make_tensor_dataset(
    train_data,
    train_conditional,
    device = device,
    scale = True
)

validation_dataset = flow.make_tensor_dataset(
    validation_data,
    validation_conditional,
    device = device,
    scale = True
)

flow.train(
    optimiser = optimiser,
    validation_dataset = validation_dataset,
    train_dataset = train_dataset,
    scheduler = scheduler,
    device = device
)


