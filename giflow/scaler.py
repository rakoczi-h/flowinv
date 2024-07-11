import numpy as np
import sklearn.preprocessing
import sklearn.decomposition
from sklearn.utils.validation import check_is_fitted

class Scaler:
    """
    Class to make scaling data easier. The data set can be scaled and compressed within this class.
    Attributes:
        scalers: list
            The scalers that are to be used for each data array that is given to the class. Each need to be a class from sklearn.preprocessing. The length of this list will be compared to the length of the data list when trying to scale.
        compressors: list
            The compressors to be used. These each need to be of type sklearn.decomposition (Default: None)
        labels: list
            The name of each data array that is to be scaled. (Default: None)
    """
    def __init__(self, scalers: list, compressors=None, labels=None):
        self.scalers = scalers
        self.compressors = compressors
        self.data_sizes = None
        self.scaled_data_sizes = None
        self.labels = labels

    def __setattr__(self, name, value):
        if name == 'compressors':
            if value is not None:
                if not isinstance(value, list):
                    raise ValueError('Expected list for compressors')
                if len(self.scalers) != len(value):
                    raise ValueError('scalers and compressors have to be the same length')
        if name == 'labels':
            if value is not None:
                if not ininstance(value, list):
                    raise ValueError('Expected list for labels')
                if len(self.scalers) != len(value):
                    raise ValueError('scalers and labels have to be the same length')
        super().__setattr__(name, value)


    def scale_data(self, data, fit=False):
        """
        Scales the given data using the scalers and compressors given in this class.
        Parameters
        ----------
            data: list
                List of data arrays to be scaled. This has to have the smae length as the self.scalers list, and self.compressors if given. 
                These list elements are scaled separately and then concatenated into an array ready to be used for training a neural network.
            fit: bool
                If True, the scaler and compressor are fitted to the data before scaling and the fitted scalers are saved in this class. If False, the data is scaled based on previously fitted scalers.
        Output
        ------
            np.ndarray
        """
        if not isinstance(data, list):
            raise ValueError("data has to be a list")
        if len(data) != len(self.scalers):
            raise ValueError("The number of scalers and data arrays are not the same.")

        if fit:
            print("Fitting scaler and compressor to data set...")
        else:
            for s in self.scalers:
                check_is_fitted(s)
            if self.compressors is not None:
                for c in self.compressors:
                    check_is_fitted(c)

        self.data_sizes = [np.shape(d) for d in data]
        data_rescaled = []
        for i, d in enumerate(data):
            if self.compressors is not None:
                print('compressing')
                if self.compressors[i] is not None:
                    if fit:
                        d = self.compressors[i].fit_transform(d)
                    else:
                        d = self.compressors[i].transform(d)
            if fit:
                d = self.scalers[i].fit_transform(d.flatten()[...,np.newaxis])
            else:
                d = self.scalers[i].transform(d.flatten()[...,np.newaxis])
            if self.compressors is not None:
                print('compressing 2')
                if self.compressors[i] is not None:
                    data_rescaled.append(np.reshape(d, (self.data_sizes[i][0], self.compressors[i].n_components_)))
                else:
                    data_rescaled.append(np.reshape(d, self.data_sizes[i]))
            else:
                data_rescaled.append(np.reshape(d, self.data_sizes[i]))
        self.scaled_data_sizes = [np.shape(d) for d in data_rescaled]
        data_rescaled = np.concatenate(data_rescaled, axis=1)
        return data_rescaled

    def inv_scale_data(self, data):
        """
        Inverse scales the fiven data using the calers and compressors given in this class.
        Parameters
        ----------
            data: array
                This data is split up based on the expected output. The input to this function is assumed to have the same shape and arrangement of data as the output of scale_data.
        Outputs
        -------
            list of unscaled data arrays
        """
        desired_shape = np.sum([self.scaled_data_sizes[i][1] for i in range(len(self.scaled_data_sizes))])

        if np.shape(data)[1] != desired_shape:
            raise ValueError('The input data is not the right shape.')
        for s in self.scalers:
            check_is_fitted(s)
        if self.compressors is not None:
            for c in self.compressors:
                check_is_fitted(c)
        data_list = []
        for sds in self.scaled_data_sizes:
            data_list.append(data[:,:sds[1]])
            data = data[:,sds[1]:]
        data = data_list
        data_unscaled = []
        for i, d in enumerate(data):
            data_shape = np.shape(d)
            d = self.scalers[i].inverse_transform(d.flatten()[..., np.newaxis])
            d = d.reshape(data_shape)
            if self.compressors is not None:
                if self.compressors[i] is not None:
                    d = self.compressors[i].inverse_transform(d)
            data_unscaled.append(d)
        return data_unscaled


