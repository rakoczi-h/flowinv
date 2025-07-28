import os
import pickle as pkl
import numpy as np
import random

from .prior import Prior
from .dataset import FaultDataset

class DataReader():
    """
    Class to contain information necessary to read in files of training a validation data for the fault inversion method.
    Parameters
    ----------
        file_names: list
            The list of filenames to read.
            The files are assumed to be .pkl and they either contain dictionaries, of FaultDataset objects.
        data_location: str
            The location from where the files are read.
        model_parameters_to_include: list
            A list of strings with each string representing a fault parameter.
        survey_coordinates_to_include: list
            The way the survey coordinates are included in the conditional. Current options are: 'x', 'y', 'z', 'x_range', 'y_range', 'z_range', 'survey_width_ratio', 'noise_scale'. Multiple can be chosen. Default: []
        noise scale: list or float
            The distribution defining the noise scale. This is directly passed as a distribution to the Prior class. If none, then no noise. Default: None
        chunk_size: int
            When the files are read in chunks, this defines the number of files within a chunk. If none, then all the files are within a chunk. Default: None
        datasize: int
            The desired number of data points in the output. If none, then it is inferred from the read files. Default: None
    """
    def __init__(self, filenames, data_location, model_info_to_include, survey_info_to_include=[], noise_scale=None, chunk_size=None, datasize=None):
        self.filenames = filenames
        self.n_files = int(len(filenames))
        self.data_location = data_location
        self.chunk_size = chunk_size
        self.model_info_to_include = model_info_to_include
        self.survey_info_to_include = survey_info_to_include
        self.noise_scale = noise_scale
        self.datasize = datasize

    def __setattr__(self, name, value):
        if name == 'filenames':
            if not isinstance(value, list):
                if isinstance(value, str):
                    value = [value]
                else:
                    raise ValueError("filenames has to be a list or a str.")
            for v in value:
                if not v[-4:] == '.pkl':
                    raise ValueError('Only .pkl files can be read.')
        super().__setattr__(name, value)

    def split_filenames(self, chunk_size=None, randomise=False):
        if chunk_size is not None:
            self.chunk_size = chunk_size
        if self.chunk_size is None:
            filenames = self.filenames.copy()
            if randomise:
                return random.shuffle(filenames)
            else:
                return filenames
        else:
            filenames = self.filenames.copy()
            if randomise:
                random.shuffle(filenames)
            else:
                filenames = self.filenames.copy()
            num_sections = int(self.n_files/self.chunk_size)
            print(f"Datasize = {self.chunk_size*num_sections}")
            filenames = [filenames[(n*self.chunk_size):((n+1)*self.chunk_size)] for n in range(num_sections)]
            return filenames

    def regenerate_noise(self, dt):
        for survey in dt.surveys:
            noise_prior = Prior(distributions={"noise_scale" : self.noise_scale})
            noise_scale = noise_prior.sample(size=1, returntype='dict')['noise_scale'][0]
            survey.noise_scale = noise_scale
            survey.make_noise()
        return dt


    def read_files(self, noise_augment=False, noise_augment_factor=2, noise_seed=None, noise_distribution=None):
        """
        Function that read the files defined by data_location and filenames in the class. This is specific for data files describing faults.               Parameters
        ----------
            noise_augment: bool
                If True, then the read data is reused with different noise realisations.
            noise_augment_factor: int
                If noise_augment is True, this is the number of times the same data is reused with different noise realisations.
        Output
        ------
            train_data: list
                Has one element, which is an ndarray with shape [dataset size, length of model_parameters_to_include].
            train_conditional: list
                Has one element, which is an ndarray with shape [dataset size, number of sh coefficients]
        """
        train_data = []
        train_conditional = []
        for f in self.filenames:
            with open(os.path.join(self.data_location, f), 'rb') as file:
                dt = pkl.load(file)
                # Reading files containing FaultDataset objects
                if isinstance(dt, FaultDataset):
                    td, tc = dt.make_data_for_network(survey_info_to_include=self.survey_info_to_include, model_info_to_include=self.model_info_to_include, add_noise=True, noise_seed=noise_seed, noise_distribution=noise_distribution)   # these objects have corresponding method to format the data that is compatible with training
                    train_data.append(td)
                    train_conditional.append(tc)
                    # Noise augmentation
                    if noise_augment:
                        if self.noise_scale is None:
                            raise ValueError('Need to provide the noise scale to the class')
                        for i in range(noise_augment_factor-1):
                            self.regenerate_noise(dt)
                            td, tc = dt.make_data_for_network(survey_info_to_include=self.survey_info_to_include, model_info_to_include=self.model_info_to_include, add_noise=True, noise_seed=noise_seed, noise_distribution=noise_distribution)
                            train_data.append(td)
                            train_conditional.append(tc)
                # Reading files containing dictionaries
                elif isinstance(dt, dict):
                    td, tc = self.read_dictionary(dt)
                    train_data.append(td)
                    train_conditional.append(tc)
                    if noise_augment:
                        for i in range(noise_augment_factor-1):
                            td, tc = self.read_dictionary(dt, make_noise=True)
                            train_data.append(td)
                            train_conditional.append(tc)
                else:
                    raise ValueError('The object type in file not supported.')

        # Reashaping the lists to the right format
        tc_list = []
        for i in range(len(train_conditional[0])):
            tc = []
            for j in range(len(train_conditional)):
                tc.append(train_conditional[j][i])
            tc = np.vstack(tc)
            tc_list.append(tc)
        tc = []
        train_conditional = tc_list

        td_list = []
        for i in range(len(train_data[0])):
            td = []
            for j in range(len(train_data)):
                td.append(train_data[j][i])
            td = np.vstack(td)
            td_list.append(td)
        td = []
        train_data = td_list

        # Cutting the data corresponding to the defined datasize
        if self.datasize is not None:
            for i, td in enumerate(train_data):
                train_data[i] = train_data[i][:self.datasize]
            for i, td in enumerate(train_conditional):
                train_conditional[i] = train_conditional[i][:self.datasize]
        else:
            self.datasize = np.shape(train_data[0])[0]

        print(f"Dataset read. Data size = {self.datasize}")
        return train_data, train_conditional

#    def read_dictionary(self, dt, make_noise=True):
#        """
#        Method to extract relevant information from a dictionary.
#        Parameters
#        ----------
#            dt: dict
#                The dictionary to read.
#            make_noise: bool
#                If True, then noise is made from the given noise_scale.
#        """
#
#        if not isinstance(dt, dict):
#            raise ValueError('Input has to be a dictionary.')
#        if not make_noise:
#            if dt['noise'] is None:
#                print('No noise in dictionary, making noise.')
#                make_noise = True
#
#        # Making noise
#        if make_noise:
#            if self.noise_scale is None:
#                raise ValueError('Either provide the noise_scale to the class, or set make_noise to False.')
#            noise_prior = Prior(distributions={"noise_scale" : self.noise_scale})
#            noise_scale_sampled = noise_prior.sample(size=(np.shape(dt['gravity'])[0],1), returntype='dict')['noise_scale']
#            gravity = []
#            for i, n in enumerate(noise_scale_sampled):
#                gravity.append(dt['gravity'][i]+np.random.normal(0.0, n, size=np.shape(dt['gravity'])[1]))
#            gravity = np.vstack(gravity)
#        # Using noise from within the dictionary
#        else:
#            gravity = []
#            for g in dt['gravity']:
#                gravity.append(g+dt['noise'][i])
#            gravity = np.vstack(gravity)
#
#        tc = [gravity]
#
#        # Making the conditional
#        if self.survey_coordinates_to_include is not None:
#            labels = ['x', 'y', 'z']
#            for idx, label in enumerate(labels):
#                if any([l==label for l in self.survey_info_to_include]):
#                    if not label in dt:
#                        raise ValueError(f"{label} can't be found in dictionary.")
#                    tc.append(dt[label])
#            labels = ['x_ranges', 'y_ranges', 'z_ranges']
#            for idx, label in enumerate(labels):
#                if any([l==label for l in self.survey_info_to_include]):
#                    if not 'survey_ranges' in dt:
#                        raise ValueError(f"survey_ranges can't be found in dictionary.")
#                    tc.append(dt['survey_ranges'][idx])
#            if any([c=='survey_width_ratio' for c in self.survey_info_to_include]):
#                if not 'survey_ranges' in dt:
#                    raise ValueError(f"survey_ranges can't be found in dictionary.")
#                survey_width_ratio = np.array([(r[0][1]-r[0][0])/(r[1][1]-r[1][0]) for r in dt['survey_ranges']])
#                survey_width_ratio = np.expand_dims(survey_width_ratio, axis=1)
#                tc.append(survey_width_ratio)
#            if any([l=='noise_scale' for l in self.survey_info_to_include]):
#                tc.append(noise_scale_sampled)
#
#        td = [np.expand_dims(dt[k], axis=1) for k in self.model_parameters_to_include]
#        return td, tc
#


