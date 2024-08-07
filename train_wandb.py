#!/scratch/wiay/2263373r/masters/conda_envs/venv/bin/python
import os
import pickle as pkl
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np
from datetime import datetime
import wandb

from giflow.scaler import Scaler
from giflow.flowmodel import FlowModel, save_flow
from giflow.read_files import read_files
from giflow.latent import FlowLatent

# Define sweep config
sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'minimize', 'name': 'val_loss'},
    'parameters':
    {
        'n_transforms': {'min': 8, 'max' : 15},
        'n_blocks_per_transform': {'min' : 1, 'max' : 5},
        'n_neurons': {'min' : 10, 'max': 80},
        'data_size': {'values': [500, 1000, 10000, 100000, 500000, 1000000]}
     }
}
# Initialize sweep by passing in config. (Optional) Provide a name of the project.
sweep_id = wandb.sweep(sweep=sweep_configuration, project='parameterised-inversion-to-present')


# ------------- Directories ---------------------------------
data_location = '/scratch/balta0/2263373r/giflow/4_paper/parameterised/' # THIS needs to be edited to give the data location

# ------------- Reading the data ----------------------------
survey_coordinates_to_include = [] # THIS needs to be edited if we want to include survey coordinates in the conditional
model_info_to_include=[]
mix_survey_order = False

datasize = 1000000 # THIS needs to be edited to give the overall desired data set size
num_files = 2 #number of files that needs to be read
train_data, train_conditional = read_files(data_location=data_location, filename='trainset', datasize=datasize, num_files=num_files, survey_coordinates_to_include=survey_coordinates_to_include, model_info_to_include=model_info_to_include, mix_survey_order=mix_survey_order)


valsize = 100000 # THIS needs to be edited to give the overall desired data set size
num_files = 1 #number of files that needs to be read
val_data, val_conditional = read_files(data_location=data_location, filename='validationset', datasize=valsize, num_files=num_files, survey_coordinates_to_include=survey_coordinates_to_include, model_info_to_include=model_info_to_include, mix_survey_order=mix_survey_order)


print(f"Data read. Location: \t {data_location}")
# ------------- Defining scalers ---------------------------
scalers = [MinMaxScaler()]
sc_data = Scaler(scalers=scalers)
sc_data.scale_data(train_data, fit=True)

scalers = [MinMaxScaler()]
sc_conditional=Scaler(scalers=scalers)

sc_conditional.scale_data(train_conditional, fit=True)

scalers = {'conditional': sc_conditional, 'data': sc_data}

# --------------- Defining the flow ------------------------
def main():
    wandb.init(project='real-box-inversion-voxelised')
    device = torch.device('cuda')
    hyperparameters={'n_inputs': 7,
                 'n_conditional_inputs': 64,
                 'n_transforms': wandb.config.n_transforms,
                 'n_blocks_per_transform': wandb.config.n_blocks_per_transform,
                 'n_neurons': wandb.config.n_neurons,
                 'batch_norm': True,
                 'batch_size': 5000,
                 'early_stopping': False,
                 'lr': 0.001,
                 'epochs': 1500
    }
    flow = FlowModel(hyperparameters=hyperparameters, datasize=datasize, scalers=scalers)
    flow.data_location = data_location
    flow.construct()

    train_size = wandb.config.data_size
    val_size = int(train_size/10)

    flowmodel = flow.flowmodel
    optimiser = torch.optim.Adam(flow.flowmodel.parameters(), lr=flow.hyperparameters['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.05, patience=100, cooldown=10,
                                                       min_lr=1e-6, verbose=True)

    train_dataset = flow.make_tensor_dataset([train_data[0][:train_size]], [train_conditional[0][:train_size]], device=device, scale=True)
    validation_dataset = flow.make_tensor_dataset([val_data[0][:val_size]], [val_conditional[0][:val_size]], device=device, scale=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hyperparameters['batch_size'], shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=hyperparameters['batch_size'], shuffle=True)

    flowmodel.to(device)
    loss_plot_freq = 10
    test_freq = 100
    # Training
    start_train = datetime.now()
    for i in range(hyperparameters['epochs']):
        start_epoch = datetime.now()

        flowmodel.train()
        train_loss = 0.0
        for batch in train_loader:
            x, y = batch
            optimiser.zero_grad()
            _loss = -flowmodel.log_prob(x, conditional=y).mean()
            _loss.backward()
            optimiser.step()
            train_loss += _loss.item()
        train_loss = train_loss / len(train_loader)

        flowmodel.eval()
        val_loss = 0.0
        for batch in validation_loader:
            x, y = batch
            with torch.no_grad():
                _loss = -flowmodel.log_prob(x, conditional=y).mean().item()
            val_loss += _loss
        val_loss = val_loss / len(validation_loader)

        wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'val_error': np.abs(val_loss-train_loss)})
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        # Testing
        if not i % test_freq and i != 0:
            print('testing')
            start_test = datetime.now()
            flowmodel.eval()
            with torch.no_grad():
                num = 2000
                z_, log_prob = flowmodel.forward_and_log_prob(validation_dataset.tensors[0][:int(num)], conditional=validation_dataset.tensors[1][:int(num)])
            print('samples drawn')
            latent_samples = z_.cpu().numpy()
            latent_logprobs = log_prob.cpu().numpy()
            latent_state = FlowLatent(latent_samples, log_probabilities=latent_logprobs)
            kl_divergence = latent_state.get_kl_divergence_statistics()
            wandb.log({'kl_div': kl_divergence['mean']})

wandb.agent(sweep_id, function=main, count=20)

