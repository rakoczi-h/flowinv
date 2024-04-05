import numpy as np
import matplotlib.pyplot as plt
import matplotlib

class GravitySurvey():
    """
    Parameters
    ----------
        ranges: list[list] [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            defines the range of the survey in the 3 different directions
            if only a single value is given for the z values, that dimension is assumed to be a constant value (no elevation changes)
        survey_coordinates: np.array
            expected shape is [num_points, 3]
        noise_scale : float
            the standard deviation of gaussian noise
        survey_shape: list
            the number of survey points on a grid, if a grid is used, or the number of survay points in total if a random arrangement is used.
    """
    def __init__(self, ranges=None, survey_coordinates=None, noise_scale=None, noise_on_location_scale=0.0, survey_shape=None):
        self.ranges = ranges
        self.survey_coordinates = survey_coordinates
        self.noise_scale = noise_scale
        self.noise_on_location_scale = noise_on_location_scale
        self.noise = None
        self.noise_on_location = None
        self.survey_shape = survey_shape
        self.gravity = None

    def __setattr__(self, name, value):
        if name == 'ranges':
            if value is not None:
                if not isinstance(value, list):
                    raise ValueError("ranges is not a list.")
                if len(value) != 3:
                    raise ValueError("The shape of ranges attribute is not as required.")
        if name == 'survey_coordinates':
            if value is not None:
                if not isinstance(value, np.ndarray):
                    raise ValueError("survey_coordinates has to be of type ndarray.")
                self.num_points = np.shape(value)[0]
        if name == 'gravity':
            if value is not None:
                if not isinstance(value, np.ndarray):
                    raise ValueError("Gravity has to be of type ndarray.")
                if self.survey_coordinates is not None:
                    if np.shape(value)[0] != np.shape(self.survey_coordinates)[0]:
                        raise ValueError("The shape of the gravity and the survey has to agree.")
        if name == 'noise':
            if value is not None:
                if not isinstance(value, np.ndarray):
                    raise ValueError("Noise has to be of type ndarray.")
        super().__setattr__(name, value)

    def make_survey(self, survey_shape=None):
        """
        Method to make the survey points.
        Parameters
        ----------
            survey_shape: list
                the type of survey configuration is inferred from the lenght of the list
                len = 1 --> randomised surve
                len = 2 --> 2D grid, [number of points in the x dimension, y dimension]
                len = 3 --> 3D grid, [x, y, z]
        """
        if self.ranges is None:
            raise ValueError('The ranges of the survey coordinates are not defined.')
        if survey_shape is not None:
            self.survey_shape = survey_shape
        x_min, x_max = self.ranges[0]
        y_min, y_max = self.ranges[1]

        self.make_noise_on_location()

        if self.survey_shape is None:
            raise ValueError('The survey_shape is not defined.')
        if len(self.survey_shape) == 2:
            z = self.ranges[2][0]
            dx = (x_max - x_min)/self.survey_shape[0]
            dy = (y_max - y_min)/self.survey_shape[1]
            x_cm = np.arange(x_min+dx/2, x_max+dx/2, dx)[:self.survey_shape[0]]
            y_cm = np.arange(y_min+dy/2, y_max+dy/2, dy)[:self.survey_shape[1]]

            X_cm, Y_cm, Z_cm = np.meshgrid(x_cm, y_cm, z, indexing='ij')
            X_cm, Y_cm, Z_cm = X_cm.ravel(), Y_cm.ravel(), Z_cm.ravel()
            survey_coordinates = np.c_[X_cm, Y_cm, Z_cm]
            survey_coordinates[:,:2] = survey_coordinates[:,:2] + self.noise_on_location[:,:2]
            self.survey_coordinates = survey_coordinates
        elif len(self.survey_shape) == 3:
            if len(self.ranges[2]) != 2:
                raise ValueError("range is not defined for the z dimension.")
            z_min, z_max = self.ranges[2]
            dx = (x_max - x_min)/self.survey_shape[0]
            dy = (y_max - y_min)/self.survey_shape[1]
            dz = (z_max - z_min)/self.survey_shape[2]
            x_cm = np.arange(x_min+dx/2, x_max+dx/2, dx)[:self.survey_shape[0]]
            y_cm = np.arange(y_min+dy/2, y_max+dy/2, dy)[:self.survey_shape[1]]
            z_cm = np.arange(z_min+dy/2, z_max+dy/2, dz)[:self.survey_shape[2]]

            X_cm, Y_cm, Z_cm = np.meshgrid(x_cm, y_cm, z_cm, indexing='ij')
            X_cm, Y_cm, Z_cm = X_cm.ravel(), Y_cm.ravel(), Z_cm.ravel()
            survey_coordinates = np.c_[X_cm, Y_cm, Z_cm]
            survey_coordinates = survey_coordinates + self.noise_on_location
            self.survey_coordinates = survey_coordinates
        elif len(self.survey_shape) == 1:
            x = np.random.uniform(self.ranges[0][0], self.ranges[0][1], size=self.survey_shape)
            y = np.random.uniform(self.ranges[1][0], self.ranges[1][1], size=self.survey_shape)
            if len(self.ranges[2]) == 2:
                z = np.random.uniform(self.ranges[2][0], self.ranges[2][1], size=self.survey_shape)
            else:
                z = np.ones(self.survey_shape)*self.ranges[2][0]
            survey_coordinates = np.vstack([x, y, z]).T
            survey_coordinates[:,:2] = survey_coordinates[:,:2] + self.noise_on_location[:,:2]
            self.survey_coordinates = survey_coordinates
        else:
            raise ValueError('The given survey_shape cannot be interpreted')
        return self.survey_coordinates

    def make_noise(self, noise_scale=None):
        if noise_scale is not None:
            self.noise_scale = noise_scale
        else:
            if self.noise_scale is None:
                raise ValueError("noise scale was not added as an input or attribute to the class.")
        num_points = self.get_number_of_surveypoints()
        self.noise = np.random.normal(loc=0.0, scale=self.noise_scale, size=num_points)
        return self.noise

    def make_noise_on_location(self, noise_on_location_scale=None):
        if noise_on_location_scale is not None:
            self.noise_on_location_scale = noise_scale
        else:
            if self.noise_on_location_scale is None:
                raise ValueError("noise scale was not added as an input or attribute to the class.")
        num_points = self.get_number_of_surveypoints()
        self.noise_on_location = np.random.normal(loc=0.0, scale=self.noise_on_location_scale, size=(num_points, 3))
        return self.noise_on_location

    def get_number_of_surveypoints(self):
        if self.survey_coordinates is not None:
            num_points = np.shape(self.survey_coordinates)[0]
        elif self.survey_shape is not None:
            num_points = 1
            for i in self.survey_shape:
                num_points = num_points*i
        else:
            raise ValueError("Not enough information is given. Provide either survey_shape or survey_coordinates to the class")
        return num_points

    def add_noise(self):
        if self.noise is None:
            raise ValueError("Noise has not been defined.")
        return self.gravity + self.noise

    def snr(self, noise_scale=None):
        """
        Compute the SNR as RMS(signal)/RMS(noise)
        """
        if noise_scale is None:
            noise_scale = self.noise_scale
            if noise_scale is None:
                raise ValueError("Set the noise_scale class attribute or function input first")
        num_points = self.get_number_of_surveypoints()
        snr = np.sqrt(np.sum((self.gravity-np.mean(self.gravity))**2)/num_points)/(noise_scale)
        return snr

    #Plotting
    def plot_pixels(self, filename='survey.png', include_noise=False):
        """
        Creates a simple pixelised image of the survey. Can only be done for gridded data.
        """
        if len(self.survey_shape) == 1:
            raise ValueError("This function is only available for gridded data points.")
        if self.gravity is None:
            raise ValueError("Compute the gravity first.")
        if include_noise is True:
            if self.noise is None:
                self.make_noise()
            plot_data = self.add_noise()
        else:
            plot_data = self.gravity
        plt.imshow(np.reshape(plot_data, self.survey_shape), extent=(np.min(self.survey_coordinates[:,0]), np.max(self.survey_coordinates[:,0]), np.max(self.survey_coordinates[:,1]), np.min(self.survey_coordinates[:,1])))
        plt.xlabel('y')
        plt.ylabel('x')
        plt.colorbar(label=r'$\mu Gal$')
        plt.savefig(filename)
        plt.close()

    def plot_contours(self, filename='survey.png', include_noise=False):
        """
        Creates a contour plot of the survey.
        """
        if self.gravity is None:
            raise ValueError("Compute the gravity first.")
        if include_noise is True:
            if self.noise is None:
                self.make_noise()
            plot_data = self.add_noise()
        else:
            plot_data = self.gravity
        levels = np.linspace(self.gravity.min(), self.gravity.max(), 7)
        cmap = 'plasma'
        norm = matplotlib.colors.Normalize(vmin=self.gravity.min(), vmax=self.gravity.max())
        fig, ax = plt.subplots()
        ax.plot(self.survey_coordinates[:,0], self.survey_coordinates[:,1], 'o', markersize=2, color='black')
        ax.tricontourf(self.survey_coordinates[:,0], self.survey_coordinates[:,1], plot_data, levels=levels, cmap=cmap, norm=norm)
        ax.set(xlim=(np.min(self.survey_coordinates[:,0]), np.max(self.survey_coordinates[:,0])), ylim=(np.min(self.survey_coordinates[:,1]), np.max(self.survey_coordinates[:,1])))
        plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=r'microGal')
        plt.savefig(filename)
        plt.close()

    def change_reference_frame(self, coordinate_centres=[0,0]):
        """
        Chenges the frame of reference of the coordiante system.
        Parameters
        ---------
            coordinate_centres: list of floats
                This should be in the current frame,
                Defines the centre point of the new coordinate system
        """
        x, y = coordinate_centres
        self.survey_coordinates[:,0] = self.survey_coordinates[:,0] - x
        self.survey_coordinates[:,1] = self.survey_coordinates[:,1] - y
        self.ranges[0][0], self.ranges[0][1] = self.ranges[0][0]-x, self.ranges[0][1]-x
        self.ranges[1][0], self.ranges[1][1] = self.ranges[1][0]-y, self.ranges[1][1]-y
        return
