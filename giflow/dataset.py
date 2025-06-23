import numpy as np
from .prior import Prior
from .survey import GravitySurvey


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
            value.setdefault("density", 1000.0)
            value.setdefault("grid_shape", None)
            value.setdefault("ranges", None)
        if name == 'survey_framework':
            if not isinstance(value, dict):
                raise ValueError("Expected dict for survey_framework.")
            value.setdefault("noise_scale", 0.0)
            value.setdefault("ranges", [[-1,1],[-1,1],[0]])
            value.setdefault("survey_shape", [10,10])
            value.setdefault("noise_on_location_scale", 0.0)
            if not isinstance(value["survey_shape"], list):
                raise ValueError("The survey shape has to be a list. Can have a single element")
        super().__setattr__(name, value)


    def make_data_for_network(self, survey_info_to_include=[], model_info_to_include=[], add_noise=True):
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
        Output
        ------
            data: list
                List of arrays with information to include in the desied data set. The first element of this list is always the model of the box (either parameterised or voxelised)
            condition: list
                List of arrays with information to include int he desired data set. The first element of this list is always the gravity array.
        """
        data = []
        if model_info_to_include:
            for key in model_info_to_include:
                arr = np.array([self.source_models[i].parameters[key] for i in range(self.size)])[...,np.newaxis]
                data.append(arr)

        conditional_gz = np.array([self.surveys[i].gravity for i in range(self.size)])
        conditional_coordinates = np.array([self.surveys[i].survey_coordinates for i in range(self.size)])

        if add_noise:
            noise = np.array([self.surveys[i].noise for i in range(self.size)])
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
            if any([l=='noise_scale' for l in survey_coordinates_to_include]):
                conditional.append(np.expand_dims(np.array([self.surveys[i].noise_scale for i in range(self.size)]), axis=1))

        return data, conditional


class FaultDataset(Dataset):
    """
    Class for making a data set of faults and corresponding gravity surveys.
    """
    def __init__(self, size: int, priors: Prior, model_framework={}, survey_framework={}):
        self.size = size
        self.priors = priors
        self.model_framework = model_framework
        self.survey_framework = survey_framework
        self.sourcemodels = None
        self.surveys = None

    def make_dataset(self):
        total_time = datetime.now()
        if parameters_dict is None:
            parameters_dict = self.priors.sample(size=self.size, returntype='dict') # if the parameters dictionary is not passed to the function, then the prior is sampled

        # Making the grid
        X = np.linspace(self.survey_framework['grid_ranges'][0][0], self.survey_framework['grid_ranges'][0][1], num=self.survey_framework['grid_resolution'])
        Y = np.linspace(self.survey_framework['grid_ranges'][1][0], self.survey_framework['grid_ranges'][1][1], num=self.survey_framework['grid_resolution'])
        X, Y = np.meshgrid(X, Y)
        X = np.expand_dims(X, axis=2)
        Y = np.expand_dims(Y, axis=2)
        Z = np.zeros(np.shape(X))
        grid = np.c_[X, Y, Z]

        # Making the survey area
        X = np.linspace(-2, 2, num=self.survey_framework['survey_shape'][0])
        Y = np.linspace(-2, 2, num=self.survey_framework['survey_shape'][1])
        X, Y = np.meshgrid(X, Y)
        Z = np.zeros(np.shape(X))
        survey_coordinates = np.c_[X.flatten(), Y.flatten(), Z.flatten()]

        grid = np.c_[X, Y, Z]
        for i in range(self.size):
            parameters = dict.fromkeys(self.model_framework['parameter_to_include'])
            parameters['density'] = self.model_framework['density']
            for key in parameters.keys():
                parameters[key] = parameters_dict[key][i]
            fault = Fault(parameters=parameters)
            fault.make_fault(grid)
            grav, _ = fault.forward_model(survey_coordinates=survey_coordinates)
            noise = np.random.normal(loc=0.0, scale=self.survey_framework['noise_scale'], size=np.shape(grav))
            grav = grav+noise

