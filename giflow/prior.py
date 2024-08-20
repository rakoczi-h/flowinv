import numpy as np
import scipy.stats
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-deep')

class Prior():
    """
    Class for defining the prior distributions.
    Parameters
    ----------
        distributions: dict of lists
            The keys in the dictionary are the parameter names.
            Each list in the dict has 3 entries. The first entry is the type of distribution, either 'Uniform' or 'Normal'
            if Uniform: ['Uniform', low limit, high limit]
            if Normal: ['Normal', mean, standard deviation]
            if the list has only 1 element and it is a float, then the value is kept at that constant
    """
    def __init__(self, distributions : dict):
        self.distributions=distributions
        self.keys=list(self.distributions.keys())

    def __setattr__(self, name, value):
        if name == 'distributions':
            self.keys = value.keys()
            for key in self.keys:
                if isinstance(value[key], list):
                    if len(value[key]) == 1:
                        if not isinstance(value[key][0], (float, int)):
                            raise ValueError("When only giving a single value for a distribution it needs to be a float or an int")
                    else:
                        if isinstance(value[key][0], str) and value[key][0] != 'Uniform' and value[key][0] != 'Normal':
                            raise ValueError('Only Uniform or Normal distributions can be given.')
                else:
                    if not isinstance(value[key], (float, int)):
                        raise ValueError("When only giving a single value for a distribution, it needs to be a float or an int")
                super().__setattr__(key, value[key])
        super().__setattr__(name, value)

    def sample(self, size, returntype='array'):
        """
        Parameters
        ----------
            size: int
                The number of samples to draw
            returntype: str
                can be 'array' or 'dict'
        """
        if returntype == 'array':
            samples = []
        elif returntype == 'dict':
            samples = dict.fromkeys(self.keys)
        else:
            raise ValueError('returntype can only be array or dict.')
        for key in self.keys:
            if isinstance(self.distributions[key], (float, int)):
                s = np.ones(size)*self.distributions[key]
            elif isinstance(self.distributions[key], list):
                if len(self.distributions[key]) == 1:
                    s = np.ones(size)*self.distributions[key][0]
                elif self.distributions[key][0] == 'Uniform':
                    s = np.random.uniform(low=self.distributions[key][1], high=self.distributions[key][2], size=size)
                elif self.distributions[key][0] == 'Normal':
                    s = np.random.normal(loc=self.distributions[key][1], scale=self.distributions[key][2], size=size)
                else:
                    raise ValueError('See documentation for right input format')
            else:
                raise ValueError('See documentation for right input format')
            if returntype == 'array':
                samples.append(s)
            elif returntype == 'dict':
                samples[key] = s
        if returntype == 'array':
            return np.array(samples).T
        elif returntype == 'dict':
            return samples

    def plot_distributions(self, filename=None):
        num = len(self.keys)
        cols = 2
        rows = int(num/2)

        samples = self.sample(size=3000, returntype='dict')

        fig, axs = plt.subplots(rows, cols)
        axs = axs.flatten()
        for i, k in enumerate(self.keys):
            axs[i].hist(samples[k], bins=100, density=True, histtype='step')
            axs[i].set_title(k)
        if filename is not None:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    def get_js_divergence(self, samples_to_compare, n=500, num_samples=2000):
        """Function calculating the Jensen-Shannon divergence between the distribution of the samples of this class and another set of samples.
        The p(x) and q(x) functions are calculated using a KDE of the input samples.
        This is done for each dimension seperately.
        Parameters
        ----------
            samples_to_compare: array
                Samples from the other sampler. [no. of samples, no. of dimensions]. Assumed to be in the original data space (not normalised)
            n: int
                The number of gridpoints to consider when computing the kdes
            num_samples: int
                The number of samples to draw from the prior.
        Output
        ------
            js: array of floats
                The list of JS-divergence values with length of the no. of parameters/dimensions.
            mean_js: float
                The mean of the js divergence values.
        """
        samples = self.sample(size=2000)
        print(np.shape(samples))
        js = []
        for i, dim in enumerate(samples.T):
            xmin = min([np.min(dim), np.min(samples_to_compare[:num_samples,i])])
            xmax = max([np.max(dim), np.max(samples_to_compare[:num_samples,i])])
            # calculate the minimum and maximum from both
            x_grid = np.arange(xmin, xmax+((xmax-xmin)/n), (xmax-xmin)/n) # the grid values we are using
            p = scipy.stats.gaussian_kde(dim)
            p_x = p.evaluate(x_grid)
            q = scipy.stats.gaussian_kde(samples_to_compare[:,i])
            q_x = q.evaluate(x_grid)
            js_pq = np.nan_to_num(np.power(jensenshannon(p_x, q_x), 2))
            js.append(js_pq)
        js = np.array(js)
        mean_js = np.mean(js)
        print(f"JS divergence statistics calculated: Mean = {mean_js}")
        return js, mean_js

