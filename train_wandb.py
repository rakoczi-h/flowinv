#!/scratch/wiay/2263373r/masters/conda_envs/venv/bin/python
import os
import pickle as pkl
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np
from datetime import datetime
import wandb
import json

from giflow.box import BoxDataset
from giflow.scaler import Scaler
from giflow.flowmodel import FlowModel, save_flow
from giflow.datareader import DataReader
from giflow.latent import FlowLatent
# Define sweep config
sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'minimize', 'name': 'val_loss'},
    'parameters':
    {
        'n_transforms': {'values': [2, 5, 12, 16]},
        'n_blocks_per_transform': {'values': [2, 5, 12, 16]},
        'n_neurons': {'values': [8, 16, 32, 64, 128]},
        'batch_size': {'values': [1000, 5000, 10000]}
     }
}
# Initialize sweep by passing in config. (Optional) Provide a name of the project.
sweep_id = wandb.sweep(sweep=sweep_configuration, project='fault-python-5-parameter')


# ------------- Directories ---------------------------------
data_location = '/scratch/balta0/2263373r/fault_python/synthetic_inversion_data/'   # THIS needs to be edited to give the data location

#with open(os.path.join(data_location, 'priors.pkl'), 'rb') as file:
#    priors = pkl.load(file)
#with open(os.path.join(data_location, 'survey_framework.json'), 'r') as file:
#    survey_framework = json.load(file)

# ------------- Defining scalers ---------------------------
model_info_to_include = ['cx', 'cy', 'l', 'alpha', 'cz']
survey_info_to_include = []
noise_distribution = Prior(distributions={'noise_scale': [0.1]})


datasize = 1000000

#if noise_scale[0] == 'LogUniform':
#    train_conditional[2] = np.log(train_conditional[2])

# Reading in files
trainsize = 500000
dr_train = DataReader(filenames=[f"trainset_{n}.pkl" for n in range(1,50)], data_location=data_location, model_info_to_include=model_info_to_include, survey_info_to_include=survey_info_to_include, datasize=trainsize)
train_data, train_conditional = dr_train.read_files(noise_distribution=noise_distribution)

# Scaling the data
sc_data = Scaler(scalers = [MinMaxScaler(), MinMaxScaler(), MinMaxScaler(), MinMaxScaler(), MinMaxScaler()]) # Need to define the scaler for each element in the train_data list.
sc_data.scale_data(train_data, fit = True) # Fit the scaler and store in the class

sc_conditional = Scaler(scalers = [MinMaxScaler()])
sc_conditional.scale_data(train_conditional, fit = True)

scalers = {'conditional': sc_conditional, 'data': sc_data}

trainsize = 1500000
dr_train = DataReader(filenames=[f"trainset_{n}.pkl" for n in range(1,150)], data_location=data_location, model_info_to_include=model_info_to_include, survey_info_to_include=survey_info_to_include, datasize=trainsize)
train_data, train_conditional = dr_train.read_files(noise_distribution=noise_distribution)

valsize = 150000
dr_train = DataReader(filenames=[f"validationset_{n}.pkl" for n in range(1,150)], data_location=data_location, model_info_to_include=model_info_to_include, survey_info_to_include=survey_info_to_include, datasize=valsize)
validation_data, validation_conditional = dr_train.read_files(noise_distribution=noise_distribution)

flow = FlowModel(scalers=scalers)
device = torch.device('cuda')
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

# --------------- Defining the flow ------------------------
def main():
    wandb.init(project='combined-inversion')

    hyperparameters={'n_inputs': 5,
                 'n_conditional_inputs': 2500,
                 'n_transforms': wandb.config.n_transforms,
                 'n_blocks_per_transform': wandb.config.n_blocks_per_transform,
                 'n_neurons': wandb.config.n_neurons,
                 'batch_norm': True,
                 'batch_size': wandb.config.batch_size,
                 'early_stopping': False,
                 'lr': 0.001,
                 'epochs': 1500
    }
    flow = FlowModel(hyperparameters=hyperparameters, datasize=trainsize, scalers=scalers)
    flow.data_location = data_location
    flow.construct()

    flowmodel = flow.flowmodel
    optimiser = torch.optim.Adam(flow.flowmodel.parameters(), lr=flow.hyperparameters['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.1, patience=100, cooldown=10,
                                                       min_lr=1e-6, verbose=True)

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
            # JS test
            #js_mean = []
            #for i in range(100):
                #with torch.no_grad():
                #    conditional = torch.repeat_interleave(torch.unsqueeze(validation_dataset.tensors[1][i], axis=0), 2000, axis=0)
                #    samples, _ = flowmodel.sample_and_log_prob(2000, conditional=conditional)
                #samples = samples.cpu().numpy()
                #js, mean_js = priors.get_js_divergence(samples, n=100, num_samples=2000, parameters_to_include=model_parameters_to_include)
                #js_mean.append(mean_js)
            #js_mean = np.mean(js_mean)
            wandb.log({'kl_div': kl_divergence['mean']})
            #wandb.log({'kl_div': kl_divergence['mean'], 'js_div': js_mean})

wandb.agent(sweep_id, function=main, count=20)

