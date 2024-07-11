import numpy as np
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
        """
        Function plotting histograms of the distributions.
        """
        num = len(self.keys)
        cols = 2
        rows = round(num/2)

        samples = self.sample(size=3000, returntype='dict')

        fig, axs = plt.subplots(rows, cols, gridspec_kw={"wspace": 0.1, "hspace": 0.9})
        axs = axs.flatten()
        print(axs)
        for i, k in enumerate(self.keys):
            axs[i].hist(samples[k], bins=100, density=True, histtype='step')
            axs[i].set_title(k)
        if filename is not None:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()
