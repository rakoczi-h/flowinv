import numpy as np

class DataSet:
    def __init__(self, size: int, priors: Priors):
        self.size = size
        self.priors = priors
        self.source_models = None
        self.surveys=None

    def make_data_for_network(self, survey_coordinates_to_include=[], model_info_to_include=[], add_noise=True, mix_survey_order=False):
        """
        Parameters
        ----------
            survey_coordinates_to_include: list
                elements can be 'x', 'y', 'z': the array of measurement coordinates are included for each data point
                            'x_range' 'y_range' 'z_range': only the range of the coordinates are included. Lower and upper limits
                            'noise_scale': the scale of the gaussian noise is included (Default: [])
            mix_survey_order: bool
                if True, the order of the survey measurement points are shuffled randomly. (Default: False)
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

        if self.model_framework['type'] == 'parameterised':
            data = np.array([self.boxes[i].parameterised_model for i in range(self.size)])
        elif self.model_framework['type'] == 'voxelised':
            data = np.array([self.boxes[i].voxelised_model for i in range(self.size)])
        data = [data]
        if any([l=='noise_scale' for l in model_info_to_include]):
            data.append(np.expand_dims(np.array([self.boxes[i].background_noise_scale for i in range(self.size)]), axis=1))

        conditional_gz = np.array([self.surveys[i].gravity for i in range(self.size)])
        conditional_coordinates = np.array([self.surveys[i].survey_coordinates for i in range(self.size)])

        if add_noise:
            noise = np.array([self.surveys[i].noise for i in range(self.size)])
            conditional_gz = conditional_gz+noise

        if mix_survey_order:
            i_arr = np.arange(np.shape(conditional_gz)[1])
            for i in range(self.size):
                np.random.shuffle(i_arr)
                conditional_gz[i,:] = conditional_gz[i,:][i_arr]
                for x in range(np.shape(conditional_coordinates)[2]):
                    conditional_coordinates[i,:,x] = conditional_coordinates[i,:,x][i_arr]

        conditional = [conditional_gz]

        labels = ['x', 'y', 'z']
        for idx, label in enumerate(labels):
            if any([l==label for l in survey_coordinates_to_include]):
                conditional.append(conditional_coordinates[:,:,idx])
        labels = ['x_ranges', 'y_ranges', 'z_ranges']
        for idx, label in enumerate(labels):
            if any([l==label for l in survey_coordinates_to_include]):
                conditional.append(np.array([self.surveys[i].ranges[idx] for i in range(self.size)]))
        if any([l=='noise_scale' for l in survey_coordinates_to_include]):
            conditional.append(np.expand_dims(np.array([self.surveys[i].noise_scale for i in range(self.size)]), axis=1))

        return data, conditional
   
