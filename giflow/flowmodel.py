import torch
import numpy as np
from glasflow.flows import RealNVP
from sklearn.model_selection import train_test_split
from datetime import datetime
import os
import json
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib
import pandas as pd
import pickle as pkl

from .latent import FlowLatent
from .plot import make_pp_plot
from .box import BoxDataset
from .scaler import Scaler

plt.style.use('seaborn-v0_8-deep')

class FlowModel():
    """
    Class making a RealNVP flow model with methods to train and test it.
    Parameters
    ----------
        hyperparameters: dict
            Defines the parameters of the flow.
            keys:
                Relating to input:
                'n_inputs': The number of inputs per data point, so the number of parameters in the data model.
                'n_conditional_inputs': The number of conditional inputs per data poitn, so the number of survey points per survey.
                Relating to flow complexity:
                'n_transforms': Number of transforms in the flow.
                'n_block_per_transform': Number of blocks per transform.
                'n_neurons': Number of neurons per block.
                Relating to the training procedure:
                'epochs': The maximum number of epochs to train for.
                'batch_norm': The batches of data are individually normalised. (default: True)
                'batch_size': The number of data points given to to flow at each train iteration.
                'early_stopping': If the validation loss stops decreasing, training is automatically stopped. (default: False)
                'lr': The learning rate. (if there is a scheduler, this will only be the original lr)
        flowmodel: glasflow.flows.RealNVP model
        datasize: int
            Number of datapoints to use for training.
        scalers: dict
            keys: 'conditional', 'data'
                The scalers can be stored here to scale the data
        save_location: str
            The directory where the flow model and its outputs are saved.
    """
    def __init__(self, hyperparameters=None, flowmodel=None, datasize=None, scalers={"conditional": None, "data": None}):
        self.flowmodel = flowmodel
        self.hyperparameters = hyperparameters
        self.datasize = datasize
        self.scalers = scalers
        self.loss = {"val": [], "train": []}
        self.save_location = ""
        self.data_location = ""

    def __setattr__(self, name, value):
        if name == 'hyperparameters':
            if value is not None:
                if not isinstance(value, dict):
                    raise ValueError('Expected dict for hyperparameters.')
                value.setdefault('batch_norm', True)
                value.setdefault('early_stopping', False)
        if name == 'save_location':
            if not value == '' and not os.path.exists(value):
                os.mkdir(value)
        if name == 'flowmodel':
            if value is not None:
                if not isinstance(value, RealNVP):
                    raise ValueError("flowmodel has to be a glasflow.flows.RealNVP object")
        super().__setattr__(name, value)

    def to_json(self):
        """
        Turns the flowmodel parameters into json file. The hyperparameters and the data size are incldued.
        """
        data = {"hyperparameters": self.hyperparameters,
                "datasize": self.datasize}
        with open(os.path.join(self.save_location, 'flow_info.json'), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def getAttributes(self):
        """
        Helper function that returns only the useful attributes of the class.
        """
        return {name: attr for name, attr in self.__dict__.items()
                if not name.startswith("__") 
                and not callable(attr)
                and not type(attr) is staticmethod}

    def load(self, location: str, device=torch.device('cuda')):
        """
        Loads a saved and trained FlowModel.
        Parameters
        ----------
            location: str
                Directory where the FlowModel.pkl and the flow.pt files are located.
            device: torch.device (defaul: 'cuda')
                The device to send the network to.
        """
        print(location)
        with open(os.path.join(location, 'FlowModel.pkl'), 'rb') as file:
            attr = pkl.load(file)
        for key in list(attr.keys()):
            self.__setattr__(key, attr[key])
        self.construct()
        self.flowmodel.load_state_dict(torch.load(os.path.join(location, 'flow.pt')))
        self.flowmodel.to(device)

    def train(self, optimiser: torch.optim, validation_dataset: torch.utils.data.TensorDataset, train_dataset: torch.utils.data.TensorDataset, scheduler=None, device=torch.device('cuda')):
        """
        The main training function, which trains, validates and plots diagnostics.
        Parameters
        ----------
            optimiser: torch.optim
                A torch optimiser object
            validation_dataset: torch.utils.data.TensorDataset
                The training dataset
            train_dataset: torch.utils.data.TensorDataset
                The validation dataset
            scheduler: torch.optim.lr_scheduler
                If provided, it is used to schedule the learning rate decay. Defaults to None.
            device: torch.device
                The device to send the flow to. Has to match that of the datasets. Defaults to cuda.
        """
        # Creating the flow
        if self.flowmodel is None:
            self.construct()

        print(f"Created flow and sent to {device}...")
        print(f"Network parameters:")
        print("----------------------------------------")
        print(f"n_inputs: \t\t {self.hyperparameters['n_inputs']}")
        print(f"n_conditional_inputs: \t {self.hyperparameters['n_conditional_inputs']}")
        print(f"n_transforms: \t\t {self.hyperparameters['n_transforms']}")
        print(f"n_blocks_per_trans: \t {self.hyperparameters['n_blocks_per_transform']}")
        print(f"n_neurons: \t\t {self.hyperparameters['n_neurons']}")
        print(f"batch_norm: \t\t {self.hyperparameters['batch_norm']}")
        print(f"batch_size: \t\t {self.hyperparameters['batch_size']}")
        print(f"optimiser: \t\t {type (optimiser).__name__}")
        print(f"scheduler: \t\t {type (scheduler).__name__}")
        print(f"early stopping: \t {self.hyperparameters['early_stopping']}")
        print(f"initial learning rate: \t {self.hyperparameters['lr']}")
        print("----------------------------------------")

        if train_dataset.tensors[0].shape[0] != self.datasize:
            raise ValueError("The size of the training data set does not agree with the desired datasize.")
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.hyperparameters['batch_size'], shuffle=True)
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=self.hyperparameters['batch_size'], shuffle=True)

        self.flowmodel.to(device)
        loss_plot_freq = 10
        test_freq = 100
        # Training
        iters_no_improve = 0
        min_val_loss = np.inf
        start_train = datetime.now()
        for i in range(self.hyperparameters['epochs']):
            start_epoch = datetime.now()
            train_loss, val_loss = self.train_iter(optimiser, validation_loader, train_loader)
            self.loss['train'].append(train_loss)
            self.loss['val'].append(val_loss)
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(self.loss['val'][-1])
                else:
                    scheduler.step()
            # Plotting the loss
            if not i % loss_plot_freq:
                self.plot_loss()
            # Testing
            if not i % test_freq and i != 0:
                torch.save(self.flowmodel.state_dict(), os.path.join(self.save_location, 'flow.pt'))
                print("----------------------------------------")
                start_test = datetime.now()
                print(f"Training time: \t {start_test-start_train}")
                print("Testing...")
                self.flowmodel.eval()
                latent_samples, latent_logprobs = self.forward_and_logprob(validation_dataset)
                latent_state = FlowLatent(latent_samples, log_probabilities=latent_logprobs)
                latent_state.get_kl_divergence_statistics()
                self.plot_flow_diagnostics(latent_state, timestamp=start_test-start_train)
                end_test = datetime.now()
                print(f"Finished testing, time taken: \t {end_test-start_test}")
                print("----------------------------------------")
            # Setting early stopping condition
            if self.loss['val'][-1] < min_val_loss:
                min_val_loss = self.loss['val'][-1]
                iters_no_improve = 0
            else:
                iters_no_improve += 1
            if self.hyperparameters['early_stopping'] and iters_no_improve == 100:
                print("Early stopping!")
                break
            end_epoch = datetime.now()
            if not i % 10:
                print(f"Epoch {i} \t train: {self.loss['train'][-1]:.3f}   \t val: {self.loss['val'][-1]:.3f}   \t t: {end_epoch-start_epoch}")
        self.flowmodel.eval()
        print('Finished training...')
        end_train = datetime.now()

        torch.save(self.flowmodel.state_dict(), os.path.join(self.save_location, 'flow.pt'))

        self.plot_loss()
        latent_samples, latent_logprobs = self.forward_and_logprob(validation_dataset)
        latent_state = FlowLatent(latent_samples, log_probabilities=latent_logprobs)
        latent_state.get_kl_divergence_statistics()
        self.plot_flow_diagnostics(latent_state, timestamp=start_test-start_train)

        print(f"Run time: \t {end_train-start_train}")

    def train_iter(self, optimiser: torch.optim, validation_loader, train_loader):
        """
        The training iteration function. A completion of one of these iteration is considered one epoch. Called in train function.
            optimiser: torch.optim
                The optimiser to use.
            validation_loader: torch.data.DataLoader
                The dataloader containing the validation data
            train_loader: torch.data.DataLoader
                The dataloader containing the training data
        """
        self.flowmodel.train()
        train_loss = 0.0
        for batch in train_loader:
            x, y = batch
            optimiser.zero_grad()
            _loss = -self.flowmodel.log_prob(x, conditional=y).mean()
            _loss.backward()
            optimiser.step()
            train_loss += _loss.item()
        train_loss = train_loss / len(train_loader)

        self.flowmodel.eval()
        val_loss = 0.0
        for batch in validation_loader:
            x, y = batch
            with torch.no_grad():
                _loss = -self.flowmodel.log_prob(x, conditional=y).mean().item()
            val_loss += _loss
        val_loss = val_loss / len(validation_loader)
        return train_loss, val_loss

    def construct(self):
        """
        Makes the RealNVP flow from the hyperparameters.
        """
        flow = RealNVP(
            n_inputs=self.hyperparameters['n_inputs'],
            n_transforms=self.hyperparameters['n_transforms'],
            n_conditional_inputs=self.hyperparameters['n_conditional_inputs'],
            n_neurons=self.hyperparameters['n_neurons'],
            n_blocks_per_transform=self.hyperparameters['n_blocks_per_transform'],
            batch_norm_between_transforms=self.hyperparameters['batch_norm'], #!
        )
        self.flowmodel = flow
        return flow

    # --------------------- Plotting Methods -------------------------------------
    def plot_loss(self):
        """
        Makes a plot of the training and validation loss wrt. number of epochs.
        """
        plt.plot(self.loss['train'], label='Train')
        plt.plot(self.loss['val'], label='Val.')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.save_location, "loss.png"))
        plt.close(
)
    def plot_flow_diagnostics(self, latent: FlowLatent, timestamp=None):
        """
        Plots diagnostics during training.
        Parameters:
            latent: FlowLatent
                Used to generate plots relating to the latent space
            timestamp:
                Any value we want to pass to be printed as an indication of timestamp. Defaults to None.
        """

        plt.figure(figsize=(20,30))
        fig, axs = plt.subplot_mosaic([['A', 'A'], ['B', 'B'], ['C', 'D']],
                                  width_ratios=np.array([1,1]), height_ratios=np.array([1,1,1]),
                                  gridspec_kw={'wspace' : -0.1, 'hspace' : 0.8})
        # Plotting the loss
        ax = axs['A']
        ax.set_box_aspect(0.2)
        ax.plot(self.loss['train'], label='Train')
        ax.plot(self.loss['val'], label='Validation')
        ax.set_xlabel('Epoch', fontdict={'fontsize': 10})
        ax.set_ylabel('Loss', fontdict={'fontsize': 10})
        if timestamp is not None:
            ax.set_title(f"Loss | {timestamp}", fontdict={'fontsize': 10})
        else:
            ax.set_title(f"Loss", fontfict={'fontsize': 10})
        ax.legend()

        # Plotting the latent space distribution
        ax = axs['B']
        for i in range(np.shape(latent.samples)[1]):
            if i == 0:
                ax.hist(latent.samples[:,i], bins=100, histtype='step', density=True, label='Latent Samples')
            else:
                ax.hist(latent.samples[:,i], bins=100, histtype='step', density=True)
        g = np.linspace(-4, 4, 100)
        ax.plot(g, norm.pdf(g, loc=0.0, scale=1.0), color='navy', label='Unit Gaussian')
        ax.set_box_aspect(0.2)
        ax.set_xlim(-4, 4)
        ax.set_title(f"Latent Space Distribution | Mean KL = {latent.kl_divergence['mean']:.3f}", fontdict={'fontsize': 10})
        ax.set_ylabel('Sample Density', fontdict={'fontsize': 10})
        ax.legend()

        # Plotting a histogram of the latent log probabilites
        ax = axs['C']
        ax.hist(latent.log_probabilities, bins=100, density=True)
        ax.set_box_aspect(1)
        ax.set_title('LS Sample Probabilities', fontdict={'fontsize': 10})
        ax.set_ylabel('Prob Density', fontdict={'fontsize': 10})
        ax.set_xlabel('Log-Prob', fontdict={'fontsize': 10})

        # Plotting an image of the correlation of the latent space samples
        ax = axs['D']
        ax.set_box_aspect(1)
        sigma = np.abs(np.corrcoef(latent.samples.T))
        im = ax.imshow(sigma, norm=matplotlib.colors.LogNorm())
        ax.set_title('LS Correlation', fontdict={'fontsize': 10})
        cbar_ax = fig.add_axes([0.8, 0.1, 0.02, 0.2]) # left, bottom, width, height
        fig.colorbar(im, cax=cbar_ax, label='corr coeff')

        plt.savefig(os.path.join(self.save_location, "diagnostics.png"), transparent=False)
        plt.close()

    # ------------------------ Drawing samples ------------------------------------
    def forward_and_logprob(self, dataset: torch.utils.data.TensorDataset, num=None):
        """
        Drawing samples from the latent space and returning their corresponding log probabilties too
        Parameters
        ----------
            dataset: tensor dataset
                The data set containg the samples which we want to forward model to the latent space.
            num: int
                the number of samples to draw (Defaults to None)
        Output
        ------
            z_: array
                Laten space samples. Same shape as data.
            log_prob: array
                The log probabilties of each sample. [no. of samples]
        """
        self.flowmodel.eval()
        if num is None:
            num = dataset.tensors[0].shape[0]
        if num > dataset.tensors[0].shape[0]:
            raise ValueError('More samples requested than data samples provided')
        with torch.no_grad():
            z_, log_prob = self.flowmodel.forward_and_log_prob(dataset.tensors[0][:int(num)], conditional=dataset.tensors[1][:int(num)])
        z_ = z_.cpu().numpy()
        log_prob = log_prob.cpu().numpy()
        return z_, log_prob

    def sample_and_logprob(self, conditional: torch.Tensor, num=1):
        """
        Drawing samples from the posterior and return their corresponding log probabilites.
        Parameters
        ----------
            conditional: torch.Tensor
                The conditional based on which we want to sample. [num_conditionals, lenght of conditional]. Can pass multiple conditionals.
            num: int
                Number of samples to draw per conditional. (Default: 1)
        """
        self.flowmodel.eval()
        if conditional.dim() == 1:
            conditional = torch.unsqueeze(conditional, dim=0)
        conditional = torch.repeat_interleave(conditional, num, axis=0)
        with torch.no_grad():
            start_sample = datetime.now()
            s, l = self.flowmodel.sample_and_log_prob(num, conditional=conditional)
            end_sample = datetime.now()
        print(f"{num} samples drawn. Time taken: \t {end_sample-start_sample}")
        s = s.cpu().numpy()

        s = self.scalers['data'].inv_scale_data(s)[0]
        l = l.cpu().numpy()
        return s, l


    # --------------------------- Testing --------------------------------------
    def pp_test(self, validation_dataset: torch.utils.data.TensorDataset, num_samples=2000, num_cases=100, num_params=10, parameter_labels=None, filename='pp_plot.png'):
        """
        Draws samples from the flow and constructs a p-p plot.
        Parameters
        ----------
            validation_dataset: torch.utils.data.TensorDataset
                The data set for which we want to compute the pp values
            num_samples: int
                Number of samples to draw for each test case (Default: 2000)
            num_cases: int
                The number of test cases to consider (Default: 100)
            num_params: int
                The number of parameters to plot (Default: 10)
            parameter_labels: list of str
                The name of the parameters. If not given, then the names will be automatically generated to be ['q1', 'q2', ...] (Default: None)
            filename: str
                The location where the image is saved. (Default: 'pp_plot.png')
        """
        truths = validation_dataset.tensors[0][:int(num_cases)].cpu().numpy()
        truths = self.scalers['data'].inv_scale_data(truths)[0]
        if np.shape(truths)[1] > num_params:
            indices = np.random.randint(np.shape(truths)[1], size=num_params)
        else:
            num_params = np.shape(truths)[1]
            indices = np.arange(0, num_params)
        if parameter_labels == None:
            parameter_labels = [f"q{x}" for x in range(num_params)] # number of parameters to get the posterior for (will be 512)
        posteriors = []
        injections = []
        with torch.no_grad():
            for cnt in range(num_cases):
                posterior = dict()
                injection = dict()
                x, _ = self.sample_and_logprob(conditional=validation_dataset.tensors[1][cnt], num=num_samples)
                for i, key in enumerate(parameter_labels):
                    posterior[key] = x[:,indices[i]]
                    injection[key] = truths[cnt,indices[i]]
                posterior = pd.DataFrame(posterior)
                posteriors.append(posterior)
                injections.append(injection)
        print("Calculated results for p-p...")
        _, pvals, combined_pvals = make_pp_plot(posteriors, injections, filename=os.path.join(self.save_location, filename))
        print("Made p-p plot...")
        return pvals, combined_pvals

    # --------------------------- Dataset ---------------------------------------
    def make_tensor_dataset(self, data, conditional, device=torch.device('cuda'), scale=True):
        """
        Makes tensor dataset.
        Parameters
        ----------
            data: np.array
                The array containing the data points. The output from the BoxDataSet.make_data_arrays() method is suitable.
            conditional: np.array
                The array containing the conditionals.
            device: torch.device
                Has to match with the one provided to train() (Default: 'cuda')
            scale: bool
                If true, the data given to the function will be scaled before it is turned into a datalaoder. (default: True)
        """
        if scale:
            if self.scalers['conditional'] is None:
                raise ValueError("The conditional scaler was not given")
            if self.scalers['data'] is None:
                raise ValueError("The data scaler was not given")
            data = self.scalers['data'].scale_data(data, fit=False)
            conditional = self.scalers['conditional'].scale_data(conditional, fit=False)
            #conditional_size = conditional.shape[0]
            #conditional = self.scalers['conditional'].transform(conditional.reshape(-1, conditional.shape[-1]))
            #conditional = conditional.reshape(conditional_size, -1)
        x_tensor = torch.from_numpy(data.astype(np.float32)).to(device)
        y_tensor = torch.from_numpy(conditional.astype(np.float32)).to(device)
        dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
        return dataset

def save_flow(flow : FlowModel):
    """
    Function to save flowmodel as a pkl file and info about it in a json.
    """
    flow.to_json()
    with open(os.path.join(flow.save_location, 'FlowModel.pkl'), 'wb') as file:
        pkl.dump(flow.getAttributes(), file)
