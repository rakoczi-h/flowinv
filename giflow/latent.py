from datetime import timedelta
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats
import pickle as pkl

class FlowLatent():
    """
    Class containing samples and statistics about the latent space of the flow.
    Parameters
    ----------
        samples: array
            Shape [no. samples, no. parameters]
            The samples drawn from the latent space
        log_probabilites: array
            The vector containing the log_probabilities corresponding to each sample (Default: None)
        kl_divergence: dict
            keys: 'mean': the mean of the kl divergence across all the dimensions. The kl-divergence here refers to the distance between a normal distribution and the latent space.
                  'std'" the standard deviation. (Default: {'mean': None, 'std': None}
    """
    def __init__(self, samples, log_probabilities=None, kl_divergence={'mean':None, 'std':None}):
        if (log_probabilities is not None):
            if (np.shape(samples)[0] != np.shape(log_probabilities)[0]):
                raise ValueError('The same number of rows are required in the samples abd log_probabilities arrays.')
            elif log_probabilities.ndim != 1:
                raise ValueError('log_probabilities has to be 1D.')
        if samples.ndim != 2:
            raise ValueError('samples has to be 2D.')
        self.samples = samples
        self.log_probabilities = log_probabilities
        self.kl_divergence = kl_divergence

    def get_kl_divergence_statistics(self, n=500):
        """Testing the Gaussianity of the latent space.
        The measure is the KL divergence between the approximate pdf of the latent space and a unit gaussian pdf.
        Parameters
        ----------
            n: int
                The number of grid points we are calculating in the comparison. (Default: 500)
        Output
        ------
            kl_divergence: dict
                keys: 'mean', 'std'
        """
        xmin = np.min(self.samples[:,0])
        xmax = np.max(self.samples[:,0])
        x_grid = np.arange(xmin, xmax+((xmax-xmin)/n), (xmax-xmin)/n) # the grid values we are using
        q = lambda x: np.exp(-x**2/2)/np.sqrt(2*np.pi) # unit gaussian pdf
        q_x = q(x_grid)
        distance = []
        for dim in self.samples.T: # looping over the latent space dimensions
            # using KDE: kernel density estimation (should trace the edges of the sample hist)
            p = scipy.stats.gaussian_kde(dim) # approximate pdf of the latent space
            p_x = p.evaluate(x_grid)
            f = p_x*np.log(p_x/q_x) # kl divergence function
            kldiv = scipy.integrate.simps(f, x_grid)
            if np.isnan(kldiv).any():
                continue
            distance.append(kldiv)
        distance = np.array(distance)
        self.kl_divergence['mean'] = np.mean(distance)
        self.kl_divergence['std'] = np.std(distance)
        print(f"KL divergence statistics calculated: Mean = {self.kl_divergence['mean']}, Std = {self.kl_divergence['std']}")
        return self.kl_divergence

    def plot_latent_hist(self, bins=100, filename='latent_samples_histogram.png'):
        """
        Plotting a histogram of latent space samples.
        Parameters
        ----------
            bins: int
                The number of bins to use in the histogram. (Default:100)
            filename: str
                Contains the path to the png file. (Default: 'latent_samples_histogram.png')
        """
        if filename[-4:] != '.png':
            raise ValueError('The filetype for filename has to be .png')
        plt.style.use('seaborn-v0_8-darkgrid')
        for col in self.samples.T:
            print(np.shape(col))
            plt.hist(col, bins, histtype='step', color='green', alpha=0.3, density=True)
        g = np.linspace(-4,4,100)
        plt.plot(g, scipy.stats.norm.pdf(g, loc=0.0, scale=1.0), color='red', label='Unit gaussian')
        plt.legend()
        if self.kl_divergence['mean'] is not None:
            plt.title(f"Latent space distribution \n Mean KL={self.kl_divergence['mean']:.3f}, std_kldiv={self.kl_divergence['std']:.3f}")
        else:
            plt.title(f"Latent space distribution")
        plt.xlim(-4, 4)
        plt.ylim(0, 1)
        plt.savefig(filename)
        plt.close()

    def plot_latent_logprob(self, filename='latent_logprobabilities.png'):
        """
        Plot a histogram of the latent space sample log probabilities.
        Parameters
        ----------
            filename: str
                The location where the image is saved. (Default: 'latent_logprobabilities.png')
        """
        if filename[-4:] != '.png':
            raise ValueError('The filetype for filename has to be .png')
        if self.log_probabilities is None:
            raise ValueError('log_probabilities has to be not None.')
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.figure(figsize=(7,7))
        plt.hist(self.log_probabilities, bins=100, color='green')
        plt.title('Latent space sample log probabilities')
        plt.savefig(filename)
        plt.close()

    def plot_latent_corr(self, filename='latent_correlation.png'):
        """
        Plot a image of the correlation matrix in the latent space.
        Parameters
        ----------
            filename: str
                The location where the image is saved. (Default: 'latent_correlation.png')
        """
        if filename[-4:] != '.png':
            raise ValueError('The filetype for filename has to be .png')

        plt.style.use('seaborn-v0_8-darkgrid')
        sigma = np.corrcoef(self.samples.T)
        plt.figure(figsize=(7,7))
        plt.imshow(sigma)
        plt.colorbar()
        plt.savefig(filename)
        plt.close()

