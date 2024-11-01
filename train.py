import os
from sklearn.preprocessing import MinMaxScaler
import torch
import pickle as pkl

from giflow.box import BoxDataset
from giflow.scaler import Scaler
from giflow.flowmodel import FlowModel, save_flow

# Defining directories
data = 'data' # where our training and validation that are located
save = 'trained_flow' # where we want to save our outputs

if not os.path.exists(save):
    os.mkdir(save)

# -------------------------------------------------
# ------------------- DATA -------------------------
# Reading in files
trainsize = 5000
with open(os.path.join(data, 'trainset.pkl'), 'rb') as file:
    dt = pkl.load(file)
    train_data, train_conditional = dt.make_data_for_network(
        survey_coordinates_to_include = ['noise_scale'],
        model_info_to_include = [],
        add_noise = True, # This refers to survey noise.
        mix_survey_order = False
    )

# Need to do this with the validation data too
valsize = 500
with open(os.path.join(data, 'valset.pkl'), 'rb') as file:
    dt = pkl.load(file)
    validation_data, validation_conditional = dt.make_data_for_network(
        survey_coordinates_to_include = ['noise_scale'],
        model_info_to_include = [],
        add_noise = True,
        mix_survey_order = False,
    )

# Scaling the data
sc_data = Scaler(scalers = [MinMaxScaler()]) # Need to define the scaler for each element in the train_data list.
sc_data.scale_data(train_data, fit = True) # Fit the scaler and store in the class

sc_conditional = Scaler(scalers = [MinMaxScaler(), MinMaxScaler()])
sc_conditional.scale_data(train_conditional, fit = True)

scalers = {'conditional': sc_conditional, 'data': sc_data}

# ------------------ FLOW --------------------------
# Defining the flow parameters
hyperparameters = {
        'n_inputs': 7, # the total number of parameters in the source model, including any additional information we chose to include
        'n_conditional_inputs': 65, # the total number of values in the conditional
        'n_transforms': 12,
        'n_blocks_per_transform': 2,
        'n_neurons': 64,
        # The parameters below define some settings for the training
        'batch_size': 5000,
        'batch_norm': True,
        'lr': 0.001,
        'epochs': 3000,
        'early_stopping': False # if set True, the training stops when the validation loss stops decreasing
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
    scheduler = None,
    device = device
)


