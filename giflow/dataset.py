import numpy as np
from datetime import datetime
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt

from .prior import Prior
from .survey import GravitySurvey
from .fault import Fault
from .utils import pad_grid


class Dataset():
    """
    General class for making a data set of source models and corresponding gravity surveys.
    Attributes
    ----------
        size: int
            Number of data points
        priors: Prior
            class that defines the priors for the fault parameters
        model_framework: dict
            keys:   type: (str) currently not used for anything
                    parameters_to_include: (list) the parameter keys that we want to include in a training dataset
                    density: (float) the density contrast between the layers separated by the faulted surface
                    grid_length: (float) the total area covered by the dispalcement model of the fault
                    grid_separation: (float) the distance between each grid point
        survey_framework: dict
            keys:   noise_scale: (float) the standard deviation of the gaussian noise added to the survey
                    sizes: (list) length 3, the size of the survey area in the three different dimensions
                    survey_shape: (list) [num_x, num_y] or [num_x, num_y, num_z] the number of survey points in eac dimension
                    fault_parameters_as_centre: (bool) if True, the survey centre is location at the fault location parameters cx cy. The effect of this is that these parameters are then defined with respect to the survey position
        sourcemodels: list
            the list of the generated source model objects (currently faults or boxes)
        surveys: list
             the list of the generated surveys
    """
    def __init__(self, size: int, priors: Prior, model_framework={}, survey_framework={}):
        self.size = size
        self.priors = priors
        self.model_framework = model_framework
        self.survey_framework = survey_framework
        self.sourcemodels = None
        self.surveys = None


    def __setattr__(self, name, value):
        if name == 'model_framework':
            if not isinstance(value, dict):
                raise ValueError("Expected dict for model_framework.")
            value.setdefault("type", 'parameterised')
            value.setdefault("noise_scale", 0.0)
            value.setdefault("shape", None)
            value.setdefault("ranges", None)
            value.setdefault("varied_parameters", None)
            value.setdefault("default_parameters", None)
        if name == 'survey_framework':
            if not isinstance(value, dict):
                raise ValueError("Expected dict for survey_framework.")
            value.setdefault("noise_scale", 0.0)
            value.setdefault("ranges", [[-1,1],[-1,1],[0]])
            value.setdefault("shape", [10,10])
            value.setdefault("noise_on_location_scale", 0.0)
            if not isinstance(value["shape"], list):
                raise ValueError("The survey shape has to be a list. Can have a single element")
        super().__setattr__(name, value)


    def make_data_for_network(self, survey_info_to_include=[], model_info_to_include=[], add_noise=True, noise_seed=None, noise_distribution=None):
        """
        Parameters
        ----------
            survey_info_to_include: list
                elements can be 'x', 'y', 'z': the array of measurement coordinates are included for each data point
                            'x_range' 'y_range' 'z_range': only the range of the coordinates are included. Lower and upper limits
            add_noise: bool
                if True, measurement noise is added to the surveys with the define noise_scale. (Default: True)
            model_info_to_include: list
                elements can be 'noise_scale': the background noise scale, used for voxelised representation. (Default: [])
            noise_seed: int
                If noise is added, each gravity survey in the conditional will have random noise added generated with this specific seed. (Default: None)
            noise_distribution: Prior
                If given, the noise added to each survey will be generated from this distribution. Has to be Prior object. (Default: None)
        Output
        ------
            data: list
                List of arrays with information to include in the desied data set. The first element of this list is always the model of the box (either parameterised or voxelised)
            conditional: list
                List of arrays with information to include int he desired data set. The first element of this list is always the gravity array.
        """
        data = []
        if model_info_to_include:
            for key in model_info_to_include:
                arr = np.array([self.sourcemodels[i].parameters[key] for i in range(self.size)])[...,np.newaxis]
                data.append(arr)

        conditional_gz = np.array([self.surveys[i].gravity for i in range(self.size)])
        conditional_coordinates = np.array([self.surveys[i].survey_coordinates for i in range(self.size)])

        if add_noise:
            noise = []
            for i in range(self.size):
                if noise_distribution:
                    noise_scale = noise_distribution.sample(size=1, returntype='array')[0]
                    self.surveys[i].noise_scale = noise_scale
                if self.surveys[i].noise is None:
                    self.surveys[i].make_noise(seed=noise_seed)
                noise.append(self.surveys[i].noise)
            conditional_gz = conditional_gz+noise

        conditional = [conditional_gz]

        if survey_info_to_include:
            labels = ['x', 'y', 'z']
            for idx, label in enumerate(labels):
                if any([l==label for l in survey_info_to_include]):
                    conditional.append(conditional_coordinates[:,:,idx])
            labels = ['x_ranges', 'y_ranges', 'z_ranges']
            for idx, label in enumerate(labels):
                if any([l==label for l in survey_info_to_include]):
                    conditional.append(np.array([self.surveys[i].ranges[idx] for i in range(self.size)]))
            if any([c=='survey_width_ratio' for c in survey_info_to_include]):
                conditional.append(np.expand_dims(np.array([((self.surveys[i].ranges[0][1]-self.surveys[i].ranges[0][0])/(self.surveys[i].ranges[1][1]-self.surveys[i].ranges[1][0])) for i in range(self.size)]), axis=1))
            if any([l=='noise_scale' for l in survey_info_to_include]):
                conditional.append(np.array([self.surveys[i].noise_scale for i in range(self.size)]))
            if any([l=='density' for l in survey_info_to_include]):
                conditional.append(np.expand_dims(np.array([self.sourcemodels[i].parameters['density'] for i in range(self.size)]), axis=1))
        return data, conditional


class FaultDataset(Dataset):
    """
    Class for making a data set of faults and corresponding gravity surveys.
    """
    def make_dataset_v2(self, parameters_dict=None, augment=True, augment_dims=['density', 'cz'], augment_num=5):
        if parameters_dict is None:
            parameters_dict = self.priors.sample(size=self.size, returntype='dict') # if the parameters dictionary is not passed to the function, then the prior is sampled
        if self.model_framework['varied_parameters'] is None:
            self.model_framework['varied_parameters'] = [key for key in parameters_dict.keys()]

        window_pad_percentage = 0.5
        # Checking conditions for the survey grid:
        if self.survey_framework['width_ratio'] is None:
            ranges = self.survey_framework['ranges']
            X = np.linspace(ranges[0][0], ranges[0][1], num=self.survey_framework['shape'][0])
            Y = np.linspace(ranges[1][0], ranges[1][1], num=self.survey_framework['shape'][1])
            X, Y = np.meshgrid(X, Y, indexing='ij')
            X = np.expand_dims(X, axis=2)
            Y = np.expand_dims(Y, axis=2)
            Z = np.zeros(np.shape(X))
            survey_coordinates = np.c_[X, Y, Z]
            change_survey = False

            # Can also make the fault grid
            pad = int(np.shape(survey_coordinates)[0]*window_pad_percentage) # padding the fault grid by 25%
            grid = pad_grid(survey_coordinates, pad, square=True)
        elif isinstance(self.survey_framework['width_ratio'], int) or isinstance(self.survey_framework['width_ratio'], float) or (isinstance(self.survey_framework['width_ratio'], list) and len(self.survey_framework['width_ratio']) == 1):
            x_size = self.survey_framework['ranges'][0][1]-self.survey_framework['ranges'][0][0]
            if (isinstance(self.survey_framework['width_ratio'], list) and len(self.survey_framework['width_ratio']) == 1):
                self.survey_framework['width_ratio'] = self.survey_framework['width_ratio'][0]
            y_size = x_size*self.survey_framework['width_ratio']
            ranges = [self.survey_framework['ranges'][0], [-y_size/2, y_size/2], self.survey_framework['ranges'][2]]
            # Making the survey grid
            X = np.linspace(ranges[0][0], ranges[0][1], num=self.survey_framework['shape'][0])
            Y = np.linspace(ranges[1][0], ranges[1][1], num=self.survey_framework['shape'][1])
            X, Y = np.meshgrid(X, Y, indexing='ij')
            X = np.expand_dims(X, axis=2)
            Y = np.expand_dims(Y, axis=2)
            Z = np.zeros(np.shape(X))
            survey_coordinates = np.c_[X, Y, Z]
            change_survey = False

            # Can also make the fault grid
            pad = int(np.shape(survey_coordinates)[0]*window_pad_percentage) # padding the fault grid by 25%
            grid = pad_grid(survey_coordinates, pad, square=True)
        else:
            change_survey = True
            x_size = self.survey_framework['ranges'][0][1]-self.survey_framework['ranges'][0][0]
            width_ratio_prior = Prior(distributions = {'width_ratio': self.survey_framework['width_ratio']})
            survey_coordinates = None
            grid = None

        self.sourcemodels = []
        self.surveys = []
        if augment:
            num_iters = int(self.size/(augment_num**len(augment_dims)))
        else:
            num_iters = self.size
        if num_iters == 0:
            num_iters = 1

        for i in range(num_iters):
            # Making the survey grid
            if change_survey is True:
                y_size = x_size*width_ratio_prior.sample(size=1, returntype='array')[0][0] # the width_ratio is defined as y/x
                ranges = [self.survey_framework['ranges'][0], [-y_size/2, y_size/2], self.survey_framework['ranges'][2]]
                # Making the survey grid
                X = np.linspace(ranges[0][0], ranges[0][1], num=self.survey_framework['shape'][0])
                Y = np.linspace(ranges[1][0], ranges[1][1], num=self.survey_framework['shape'][1])
                X, Y = np.meshgrid(X, Y, indexing='ij')
                X = np.expand_dims(X, axis=2)
                Y = np.expand_dims(Y, axis=2)
                Z = np.zeros(np.shape(X))
                survey_coordinates = np.c_[X, Y, Z]

                # Making padded fault grid
                pad = int(np.shape(survey_coordinates)[0]*window_pad_percentage) # padding the fault grid by 25%
                grid = pad_grid(survey_coordinates, pad, square=True)

            # Making the fault
            parameters = {}
            if self.model_framework['default_parameters']:
                for key in self.model_framework['default_parameters']:
                    parameters[key] = self.model_framework['default_parameters'][key] # first the default values are loaded into the dict
            for key in self.model_framework['varied_parameters']:
                parameters[key] = parameters_dict[key][i] # then the varied values are added
            fault = Fault(parameters=parameters)


            # Generating the fault
            fault.make_fault(grid=grid)
            fault.displacement_profile[fault.displacement_profile>fault.parameters['cz']] = fault.parameters['cz']
            # Computing the forward model
            pad = np.shape(grid)[0]
            window_width = 0.1
            if augment:
                if any([l=='density' for l in augment_dims]):
                    densities = parameters_dict['density'][i*augment_num:(i*augment_num+augment_num)]
                else:
                    densities = None
                if any([l=='cz' for l in augment_dims]):
                    depths = parameters_dict['cz'][i*augment_num:(i*augment_num+augment_num)]
                else:
                    depths = None
                _, k_mag, R1, grid = fault.forward_model(remove_min=True, num_components=100, zero_pad=True, pad_width=[pad, pad],  win=('tukey', window_width))
                gzs, depths, densities = fault.forward_from_fourier(k_mag, R1, grid, densities=densities, depths=depths, survey_coordinates=survey_coordinates, remove_min=True)
                for j, gz in enumerate(gzs):
                    fault.parameters['density'] = densities[j]
                    fault.parameters['cz'] = depths[j]

                    survey = GravitySurvey(gravity=gz.flatten(), ranges=ranges, shape=self.survey_framework['shape'])
                    self.sourcemodels.append(Fault(parameters=fault.parameters.copy())) # addig a copy of the fault object, with only its parameters
                    self.surveys.append(survey)
            else:
                gz, _, _, _ = fault.forward_model(survey_coordinates=survey_coordinates, remove_min=True, num_components=100, zero_pad=True, pad_width=[pad, pad],  win=('tukey', window_width))
                fault.displacement_profile = None
                fault.grid = None
                survey = GravitySurvey(gravity=gz.flatten(), ranges=ranges, shape=self.survey_framework['shape'])
                self.sourcemodels.append(fault)
                self.surveys.append(survey)
        self.sourcemodels = self.sourcemodels[:self.size]
        self.surveys = self.surveys[:self.size]
        return self.sourcemodels, self.surveys

