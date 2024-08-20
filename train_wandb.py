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
        'n_transforms': {'min': 1, 'max' : 15},
        'n_blocks_per_transform': {'min' : 1, 'max' : 10},
        'n_neurons': {'min' : 10, 'max': 80},
        'batch_size': {'min' : 1000, 'max': 10000}
     }
}
# Initialize sweep by passing in config. (Optional) Provide a name of the project.
sweep_id = wandb.sweep(sweep=sweep_configuration, project='combined-inversion')


# ------------- Directories ---------------------------------
data_location = '/scratch/balta0/2263373r/giflow/box/narrow_volume/combined/' # THIS needs to be edited to give the data location

# ------------- Reading the data ----------------------------
survey_coordinates_to_include = ['x', 'y', 'noise_scale'] # THIS needs to be edited if we want to include survey coordinates in the conditional
model_info_to_include=[]
mix_survey_order = False

datasize = 1000000 # THIS needs to be edited to give the overall desired data set size
train_data, train_conditional = read_files(data_location=data_location, filenames=['trainset_0.pkl', 'trainset_1.pkl'], datasize=datasize, survey_coordinates_to_include=survey_coordinates_to_include, model_info_to_include=model_info_to_include, mix_survey_order=mix_survey_order)


valsize = 100000 # THIS needs to be edited to give the overall desired data set size
val_data, val_conditional = read_files(data_location=data_location, filenames=['validationset_0.pkl'], datasize=valsize, survey_coordinates_to_include=survey_coordinates_to_include, model_info_to_include=model_info_to_include, mix_survey_order=mix_survey_order)

# ------------- Defining the prior ---------------------
with open(os.path.join(data_location, "validationset_0.pkl"), 'rb') as file:
    dt_val = pkl.load(file)
priors = dt_val.priors

print(f"Data read. Location: \t {data_location}")
# ------------- Defining scalers ---------------------------
scalers = [MinMaxScaler()]
sc_data = Scaler(scalers=scalers)
sc_data.scale_data(train_data, fit=True)

scalers = [MinMaxScaler(), MinMaxScaler(), MinMaxScaler(), MinMaxScaler()]
sc_conditional=Scaler(scalers=scalers)
sc_conditional.scale_data(train_conditional, fit=True)

scalers = {'conditional': sc_conditional, 'data': sc_data}

# --------------- Defining the flow ------------------------
def main():
    wandb.init(project='combined-inversion')
    device = torch.device('cuda')
    hyperparameters={'n_inputs': 7,
                 'n_conditional_inputs': 193,
                 'n_transforms': wandb.config.n_transforms,
                 'n_blocks_per_transform': wandb.config.n_blocks_per_transform,
                 'n_neurons': wandb.config.n_neurons,
                 'batch_norm': True,
                 'batch_size': wandb.config.batch_size,
                 'early_stopping': False,
                 'lr': 0.001,
                 'epochs': 1500
    }
    flow = FlowModel(hyperparameters=hyperparameters, datasize=datasize, scalers=scalers)
    flow.data_location = data_location
    flow.construct()

    train_size = datasize
    val_size = valsize

    flowmodel = flow.flowmodel
    optimiser = torch.optim.Adam(flow.flowmodel.parameters(), lr=flow.hyperparameters['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.05, patience=100, cooldown=10,
                                                       min_lr=1e-6, verbose=True)

    train_dataset = flow.make_tensor_dataset(train_data, train_conditional, device=device, scale=True)
    validation_dataset = flow.make_tensor_dataset(val_data, val_conditional, device=device, scale=True)
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
            js_mean = []
            for i in range(100):
                with torch.no_grad():
                    conditional = torch.repeat_interleave(torch.unsqueeze(validation_dataset.tensors[1][i], axis=0), 2000, axis=0)
                    samples, _ = flowmodel.sample_and_log_prob(2000, conditional=conditional)
                samples = samples.cpu().numpy()
                js, mean_js = priors.get_js_divergence(samples, n=100, num_samples=2000)
                js_mean.append(mean_js)
            js_mean = np.mean(js_mean)
            wandb.log({'kl_div': kl_divergence['mean'], 'js_div': js_mean})

wandb.agent(sweep_id, function=main, count=30)

