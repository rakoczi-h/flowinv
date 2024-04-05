import numpy as np
import itertools
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from .utils import rotate
from .prior import Prior
from .survey import GravitySurvey

class Box:
    """
    Class that creates box objects for gravity anaylsis.
    Parameters
    ----------
        parameters: dict
            The parameters defining the box.
        density: float
            The density contrast between the box and the background
        parameterised_model: np.array
            The parameterised model for the box. If parameters are given, it is automatically constructed.
        voxelised_model: np.array
            The voxelised model of the box. An array of density values.
        voxel_grid: np.array
            The grid that the densities are defined on.
        background_noise_scale: float
            The scale of the gaussian noise added to simulate background effects.
    """
    def __init__(self, parameters=None, density=None, parameterised_model=None, voxelised_model=None, background_noise_scale=0.0, parameter_labels=None):
        self.density = density
        self.voxelised_model = voxelised_model
        self.parameterised_model = parameterised_model
        self.background_noise_scale=background_noise_scale
        self.parameter_labels = parameter_labels
        self.parameters = parameters # 
        self.voxel_grid = None
        self.ranges = None
        self.grid_shape = None

    def __setattr__(self, name, value):
        if name == 'parameters':
            if value is not None:
                if not isinstance(value, dict):
                    raise ValueError("parameters has to be a dictionary")
                if self.parameter_labels is None:
                    self.parameter_labels = list(value.keys())
                for i, key in enumerate(self.parameter_labels):
                    super().__setattr__(key, value[key])
        super().__setattr__(name, value)

    def forward_model(self, survey_coordinates, model_type='parameterised'):
        """
        Computes the forward model for the box.
        Parameters
        ----------
            survey: np.array
                has to contain the coordinates of the x y z survey locations [number of survey points, 3]
            model_type: str
                Define which model type to compute the forward model from. Can be parameterised or voxelised.
        """
        if model_type=='parameterised':
            if self.parameters is None:
                raise ValueError('The parameters are not defined. Either define the parameters of set from_voxelised to True.')

            survey_coordinates[:,:2] = rotate(survey_coordinates[:,:2], origin=(self.px, self.py), angle=-self.alpha)
            limits = np.expand_dims(np.array([[self.px-self.lx/2,self.px+self.lx/2], [self.py-self.ly/2,self.py+self.ly/2], [self.pz-self.lz/2,self.pz+self.lz/2]]), axis=0)
            gz = self.get_gz(limits=limits, densities=self.density, survey_coordinates=survey_coordinates)
        elif model_type=='voxelised':
            if self.voxel_grid is None:
                raise ValueError("The voxel_grid is not defined")
            if self.voxelised_model is None:
                self.translate_to_voxels()
            gz = self.get_gz(limits=self.voxel_grid, densities=self.voxelised_model, survey_coordinates=survey_coordinates)
        else:
            raise ValueError('model_type can only be parameterised or voxelised.')
        return gz

    def get_gz(self, limits, densities, survey_coordinates):
        """
        Jung (1961) and Plouff (1966), review Li (1998)
        A vectorised version of get_gz_analytical. Calculates the gravitational signature due to multiple boxes,
        and sums the values in the end. Useful for voxelised models.
        Parameters
        ----------
            limits: array
                [number of voxels, coordinates, limits] coordinate order is xyz, and limits are 2, min and max
            survey_coordinates: array
                [number of points, coordinates], coordinate order is xyz
            densities: array or float
                vector of floats with the same size as the number of limits defined.
        output: gz: [number of points] array of gravity values in microGal, at the given survey locations
        """
        if isinstance(densities, float):
            densities = np.array([densities])
        if np.shape(densities)[0] != np.shape(limits)[0]:
            print("Sizes of density and voxel vectors do not match.")
            exit()
        G = 6.6743e-11
        survey_loc = np.repeat(np.expand_dims(survey_coordinates, axis=0), np.shape(limits)[0], axis=0) #[num_voxels, num_surveys, 3]
        limits = np.repeat(np.expand_dims(limits, axis=1), np.shape(survey_loc)[1], axis=1) # [num_voxels, num_surveys, 3, 2]
        xs = [survey_loc[:,:,0]-limits[:,:,0,0], survey_loc[:,:,0]-limits[:,:,0,1]] #[2, num_voxels, num_surveys]
        ys = [survey_loc[:,:,1]-limits[:,:,1,0], survey_loc[:,:,1]-limits[:,:,1,1]]
        zs = [survey_loc[:,:,2]-limits[:,:,2,0], survey_loc[:,:,2]-limits[:,:,2,1]]
        c = [1,2]
        coefs = np.array(list(itertools.product(c,c,c)))
        mu = np.power(-1.0, coefs[:,0])*np.power(-1.0, coefs[:,1])*np.power(-1.0, coefs[:,2])
        mu = np.repeat(np.expand_dims(mu, axis=1), np.shape(xs)[2], axis=1)
        mu = np.repeat(np.expand_dims(mu, axis=1), np.shape(xs)[1], axis=1) # [8, num_voxels, num_surveys]
        terms = np.array(list(itertools.product(xs, ys, zs))) # [8, 3, num_voxels, num_surveys]
        r = np.sqrt(terms[:,0,:,:]**2+terms[:,1,:,:]**2+terms[:,2,:,:]**2) # [8, num_voxels, num_surveys]
        # fixing zeros
        logterm1 = terms[:,1,:,:]+r
        logterm1[np.where(logterm1==0)]=1e-100
        logterm2 = terms[:,0,:,:]+r
        logterm2[np.where(logterm2==0)]=1e-100
        divterm = terms[:,2,:,:]*r
        divterm[np.where(divterm==0)]=1e-100
        eq = mu*(terms[:,0,:,:]*np.log(logterm1) + terms[:,1,:,:]*np.log(logterm2) - terms[:,2,:,:]*np.arctan((terms[:,0,:,:]*terms[:,1,:,:])/(divterm)))
        rho = np.repeat(np.expand_dims(densities, axis=1), np.shape(xs)[2], axis=1)
        gz = -G*rho*np.sum(eq, axis=0)*1e8 # in microGal, summing the terms in the equation for a single voxel
        gz = np.sum(gz, axis=0) # summing the contribution from each voxel
        return gz

    def translate_to_voxels(self, voxel_grid=None, background_noise_scale=None, density=None):
        """Makes a rotated box with the given parameters [px, py, pz, lx, ly, lz, alpha_x, alpha_y] translated
        to density values at the voxel grid location specified.
        The background density is assumed to be 0.
        Parameters
        ----------
            voxel_grid: array
                Needs to contain the boundaries of the voxels. [number of voxels, 3, 2]
                Coordinate order is xyz given in the 2nd dimension. Limits are 2, min and max, given in the 3rd dimension
            density: float
                Contains a set of parameters that define a single box
            background_noise_scale: float
                The standard deviation of the background noise that is added to the model. If None, the value stored in the class is used.
                CAUTION: overwrites previous value if given.
        Output
        ------
            model: array
                The voxelised representation of the box defined on the input grid and parameters. [number of voxels]
        """
        if voxel_grid is not None:
            self.voxel_grid = voxel_grid
        else:
            if self.voxel_grid is None:
                raise ValueError("Voxel grid is not given as an input or an attribute of the class.")

        if background_noise_scale is not None:
            self.background_noise_scale = background_noise_scale

        if density is not None:
            self.density = density
        else:
            if self.density is None:
                raise ValueError('the density is not given as an inout or an attribute to the class.')

        num_voxels = np.shape(self.voxel_grid)[0]
        rho_obj = self.density # the relative density within the object
        rho_bg = 0.0
        bg = np.zeros(num_voxels)
        if self.background_noise_scale != 0.0:
            bg = bg + np.random.normal(loc=0.0, scale=self.background_noise_scale, size=np.shape(bg)[0])
        # calculating the locations of the vertices of the rectangle, which is the box when viwed from above
        p1 = tuple(rotate(np.array([self.px-self.lx/2, self.py+self.ly/2]), origin=(self.px,self.py), angle=self.alpha))
        p2 = tuple(rotate(np.array([self.px+self.lx/2, self.py+self.ly/2]), origin=(self.px,self.py), angle=self.alpha))
        p3 = tuple(rotate(np.array([self.px-self.lx/2, self.py-self.ly/2]), origin=(self.px,self.py), angle=self.alpha))
        p4 = tuple(rotate(np.array([self.px+self.lx/2, self.py-self.ly/2]), origin=(self.px,self.py), angle=self.alpha))
        # making the 1d rectangle
        rect1 = Polygon([p1, p2, p4, p3])
        # defining the extent of the box in the z direction
        z_top = self.pz+self.lz/2
        z_bottom = self.pz-self.lz/2
        densities = []
        for i in range(np.shape(self.voxel_grid)[0]): # looping over each ovxel
            q1 = [self.voxel_grid[i,0,0], self.voxel_grid[i,1,0]] # (x1, y1)
            q2 = [self.voxel_grid[i,0,0], self.voxel_grid[i,1,1]] # (x1, y2)
            q3 = [self.voxel_grid[i,0,1], self.voxel_grid[i,1,0]] # (x2, y1)
            q4 = [self.voxel_grid[i,0,1], self.voxel_grid[i,1,1]] # (x2, y2)
            z1 = self.voxel_grid[i,2,1] # the limits of the voxel in z, z1 is the max
            z2 = self.voxel_grid[i,2,0]
            width = z1-z2
            area_voxel = (self.voxel_grid[i,0,1]-self.voxel_grid[i,0,0])*(self.voxel_grid[i,1,1]-self.voxel_grid[i,1,0])
            if z1 <= z2:
                 print('Gridheights not well defined')
                 exit()
            # making the 1d rectangle of the voxel, when viewed from above
            rect2 = Polygon([q1, q2, q4, q3])
            # we are assuming here that the minimum size of the air volume in z is the size of a voxel in the density space
            # checking overlaps in the z dimension
            if z1 <= z_top and z2 >= z_bottom:
                z_ratio = 1
            elif z1 > z_top and z2 < z_top:
                z_ratio = (z_top-z2)/(width) # defining the amount by which the voxel overlaps with the box, only considering the z direction first
            elif z2 < z_bottom and z_bottom < z1:
                z_ratio = (z1-z_bottom)/(width)
            else:
                z_ratio = 0
            # checking whether the two defined rectangles overlap, and by how much
            if rect1.intersects(rect2):
                intersection = rect1.intersection(rect2)
                rel_area = intersection.area/area_voxel
                dens = rho_obj*rel_area*z_ratio+rho_bg*(1-rel_area)*(1-z_ratio)
                densities.append(dens)
            else:
                densities.append(rho_bg) # if no overlap, the added density at the specific voxel loction is just 0
        densities = np.array(densities)
        self.voxelised_model = bg+densities
        return self.voxelised_model, self.voxel_grid

    def translate_to_parameterised_model(self, keys=None):
        if keys is None:
            keys = self.parameter_labels
        self.parameterised_model = np.array([self.parameters[key] for key in keys])
        return self.parameterised_model

    def translate_to_parameters(self):
        if self.parameterised_model is None:
            raise ValueError("Need to define parameterised model")
        if self.parameter_labels is None:
            raise ValueError("Need to define parameter labels")
        parameters = dict.fromkeys(self.parameter_labels)
        for idx, key in enumerate(self.parameter_labels):
            parameters[key] = self.parameterised_model[idx]
        self.parameters = parameters

    def make_voxel_grid(self, grid_shape, ranges):
        """Creates a grid of voxel locations for a rectangular volume
        Parameters
        ----------
            grid_shape: list
                [nx, ny, nz]: The number of voxels along each direction
            ranges: list
                [[xmin, xmax], [ymin, ymax], [zmin, zmax]]: The coords of the limits of the voxelised area
        Output
        ------
            coords: array
                The coordinates of the voxel limits [number of voxels, 3, 2]
                The last dimension is the high and low limits, the 2nd dimension is x y z
        """
        self.ranges = ranges
        self.grid_shape = grid_shape
        dx = (self.ranges[0][1]-self.ranges[0][0])/self.grid_shape[0]
        dy = (self.ranges[1][1]-self.ranges[1][0])/self.grid_shape[1]
        dz = (self.ranges[2][1]-self.ranges[2][0])/self.grid_shape[2]
        x_min = np.arange(self.ranges[0][0], self.ranges[0][1], dx)
        y_min = np.arange(self.ranges[1][0], self.ranges[1][1], dy)
        z_min = np.arange(self.ranges[2][0], self.ranges[2][1], dz)
        X_min, Y_min, Z_min = np.meshgrid(x_min, y_min, z_min, indexing='xy') # CHNAGE BACK To ij
        X_min, Y_min, Z_min = X_min.ravel(), Y_min.ravel(), Z_min.ravel()
        coords_min = np.c_[X_min, Y_min, Z_min]
        coords_min = np.expand_dims(coords_min, axis=2)

        x_max = np.arange(self.ranges[0][0]+dx, self.ranges[0][1]+dx, dx)
        y_max = np.arange(self.ranges[1][0]+dy, self.ranges[1][1]+dy, dy)
        z_max = np.arange(self.ranges[2][0]+dz, self.ranges[2][1]+dz, dz)
        X_max, Y_max, Z_max = np.meshgrid(x_max, y_max, z_max, indexing='xy')
        X_max, Y_max, Z_max = X_max.ravel(), Y_max.ravel(), Z_max.ravel()
        coords_max = np.c_[X_max, Y_max, Z_max]
        coords_max = np.expand_dims(coords_max, axis=2)
        self.voxel_grid = np.concatenate((coords_min, coords_max), axis=2)
        return self.voxel_grid

    #Plotting
    def plot_voxel_slices(self, filename='voxel_slices.png', slices_per_dimension=3):
        """
        Takes 3 slices of the voxelised model in all 3 dimensions and displas the voxel values in that slice.
        Parameters:
        ----------
            slices_per_dimension: int
                The number of slices to pick per dimension, default is 3
                Cannot be more than the number of voxels in that dimension.
        """
        if self.voxelised_model is None:
            raise ValueError("The voxelisd_model is not constructed")
        vmin = np.min(self.voxelised_model)
        vmax = np.max(self.voxelised_model)
        box_reshaped = np.reshape(self.voxelised_model, self.grid_shape, order='C')
        if any([i < slices_per_dimension for i in self.grid_shape]):
            raise ValueError("More slices requested than voxels in at least one of the dimensions.")
        slices_x = np.arange(0, self.grid_shape[0], self.grid_shape[0]/slices_per_dimension, dtype=int)
        slices_y = np.arange(0, self.grid_shape[1], self.grid_shape[1]/slices_per_dimension, dtype=int)
        slices_z = np.arange(0, self.grid_shape[2], self.grid_shape[2]/slices_per_dimension, dtype=int)
        slices = [slices_x, slices_y, slices_z]
        extent_x = np.min(self.voxel_grid[1])
        # slices in the x dimension
        for j in range(slices_per_dimension):
            for i in range(3):
                plt.subplot(slices_per_dimension,3,(i+1)+(j*3))
                if i == 0:
                    if j == 0:
                        plt.title('X Slices')
                    if j == 2:
                        plt.xlabel('y')
                        plt.ylabel('z')
                    plot_data = box_reshaped[slices[i][j],:,:].T
                    extent = (np.min(np.mean(self.voxel_grid[:,1,:], axis=1)), np.max(np.mean(self.voxel_grid[:,1,:], axis=1)), np.min(np.mean(self.voxel_grid[:,2,:], axis=1)), np.max(np.mean(self.voxel_grid[:,2,:], axis=1)))
                elif i == 1:
                    if j == 0:
                        plt.title('Y Slices')
                    if j == 2:
                        plt.xlabel('x')
                        plt.ylabel('z')
                    plot_data = box_reshaped[:,slices[i][j],:].T
                    extent = (np.min(np.mean(self.voxel_grid[:,0,:], axis=1)), np.max(np.mean(self.voxel_grid[:,0,:], axis=1)), np.min(np.mean(self.voxel_grid[:,2,:], axis=1)), np.max(np.mean(self.voxel_grid[:,2,:], axis=1)))
                elif i == 2:
                    if j == 0:
                        plt.title('Z Slices')
                    if j == 2:
                        plt.xlabel('x')
                        plt.ylabel('y')
                    plot_data = box_reshaped[:,:,slices[i][j]].T
                    extent = (np.min(np.mean(self.voxel_grid[:,0,:], axis=1)), np.max(np.mean(self.voxel_grid[:,0,:], axis=1)), np.max(np.mean(self.voxel_grid[:,1,:], axis=1)), np.min(np.mean(self.voxel_grid[:,1,:], axis=1)))
                plt.imshow(plot_data, vmin=vmin, vmax=vmax, extent=extent, aspect='equal')
        plt.savefig(filename)
        plt.close()

    def intact_plot_3D_mesh(self, axis_limits=None, filename='3D_mesh.html'):
        """
        Makes a 3D image of the box from its parameters.
        Parameters:
            axis_limits: array
                np.array([[xmin, xmax], [ymin,ymax], [zmin,zmax]])
                if None, it is infered from the voxel_grid or is set to a default value
        """
        # define the ranges of the coordinates
        x1 = self.px - self.lx/2
        x2 = self.px + self.lx/2
        y1 = self.py - self.ly/2
        y2 = self.py + self.ly/2
        z1 = self.pz - self.lz/2
        z2 = self.pz + self.lz/2
        x_arr = np.arange(x1, x2, (x2-x1)/10) # 10 points in each grid
        y_arr = np.arange(y1, y2, (y2-y1)/10) # 10 points in each grid
        z_arr = np.arange(z1, z2, (z2-z1)/10) # 10 points in each grid
        # make grids of points for the faces
        arrs = [x_arr, y_arr, z_arr]
        lims = [[x1, x2], [y1, y2], [z1, z2]]
        points = []
        for i in range(3):
            arrs[0] = x_arr
            arrs[1] = y_arr
            arrs[2] = z_arr
            for j in range(2):
                arrs[i] = lims[i][j]
                x, y, z = np.meshgrid(arrs[0], arrs[1], arrs[2], indexing='ij')
                x, y, z = x.ravel(), y.ravel(), z.ravel()
                coords = np.c_[x, y, z]
                points.append(coords)
        points = np.vstack(points)
        points[:, :2] = rotate(points[:, :2], origin=(self.px, self.py), angle=self.alpha)
        # defining the axis limits
        if axis_limits is None:
            if self.voxel_grid is None:
                axis_limits = np.array([[-100,100],[-100,100],[-100,100]])
            else:
                xmin = np.min(np.mean(self.voxel_grid[:,0,:], axis=1))
                xmax = np.max(np.mean(self.voxel_grid[:,0,:], axis=1))
                ymin = np.min(np.mean(self.voxel_grid[:,1,:], axis=1))
                ymax = np.max(np.mean(self.voxel_grid[:,1,:], axis=1))
                axis_limits = np.array([[xmin, xmax],[ymin, ymax],[zmin, zmax]])
        else:
            axis_limits = axis_limits
        # plotting
        fig = go.Figure(data=go.Mesh3d(x=points[:,0], y=points[:,1], z=points[:,2],
                        alphahull=0, # ideal fox convex bodies
                        opacity = 0.4,
                        color = "rgb(49, 5, 151)"))
        fig.update_layout(height = 800,
                          width = 1000,
                          font = dict(size=12),
                          scene = dict(
                               xaxis = dict(range=[axis_limits[0][0], axis_limits[0][1]],
                                            backgroundcolor="rgb(226, 183, 249)",
                                            gridcolor="white",
                                            showbackground=True,
                                            zerolinecolor="white",),
                               yaxis = dict(range=[axis_limits[1][0], axis_limits[1][1]],
                                            backgroundcolor="rgb(249, 173, 182)",
                                            gridcolor="white",
                                            showbackground=True,
                                            zerolinecolor="white",),
                               zaxis = dict(range=[axis_limits[2][0], axis_limits[2][1]],
                                            backgroundcolor="rgb(247, 238, 166)",
                                            gridcolor="white",
                                            showbackground=True,
                                            zerolinecolor="white",),
                               xaxis_title='x [m]',
                               yaxis_title='y [m]',
                               zaxis_title='z [m]',
                               aspectmode='manual',
                               aspectratio=dict(x=(axis_limits[0][1]-axis_limits[0][0])/(axis_limits[0][1]-axis_limits[0][0]),
                                                y=(axis_limits[1][1]-axis_limits[1][0])/(axis_limits[0][1]-axis_limits[0][0]),
                                                z=(axis_limits[2][1]-axis_limits[2][0])/(axis_limits[0][1]-axis_limits[0][0]))))
        fig.add_annotation(text=r'x = {px:.2f} m <br>y = {py:.2f} m <br>z = {pz:.2f} m <br>l = {lx:.2f} m <br>w = {ly:.2f} m <br>d = {lz:.2f} m <br>angle = {alpha:.2f} rad'.format(px=self.px, py=self.py, pz=self.pz, lx=self.lx, ly=self.ly, lz=self.lz, alpha=self.alpha),
                        font = dict(size=12),
                        align='left',
                        showarrow=False,
                        xref='paper',
                        yref='paper',
                        x=1.0,
                        y=0.7,
                        bordercolor='black',
                        borderwidth=1)
        if filename[-5:] == '.html':
            fig.write_html(filename)
        elif filename[-4:] == '.png':
            fig.write_image(filename)
        else:
            raise ValueError("Only .html and .png file extensions are allowed")
        plt.close()

class BoxDataset:
    def __init__(self, size: int, priors: Prior, model_framework={}, survey_framework={}):
        self.size = size
        self.priors = priors
        self.model_framework = model_framework
        self.survey_framework = survey_framework
        self.parameter_labels = priors.keys
        self.boxes = None
        self.surveys = None

    def __setattr__(self, name, value):
        if name == 'model_framework':
            if not isinstance(value, dict):
                raise ValueError("Expected dict for model_framework.")
            value.setdefault("type", 'parameterised')
            value.setdefault("noise_scale", 0.0)
            value.setdefault("density", -1600.0)
            value.setdefault("grid_shape", None)
            value.setdefault("ranges", None)
        if name == 'survey_framework':
            if not isinstance(value, dict):
                raise ValueError("Expected dict for survey_framework.")
            value.setdefault("noise_scale", 0.0)
            value.setdefault("ranges", [[-10,10],[-10,10],[0]])
            value.setdefault("survey_shape", [5,5])
            value.setdefault("noise_on_location_scale", 0.0)
            if not isinstance(value["survey_shape"], list):
                raise ValueError("The survey shape has to be a list. Can have a single element")
        super().__setattr__(name, value)

    def make_dataset(self, parameters_dict=None):
        """
        Makes a dataset of boxes.
        Parameters
        ----------
            parameters_dict: dict
                If None, then the box parameters are randomly generated from the prior.
                Otherwise the parameters are taken from the dictionary
        """
        if parameters_dict is None:
            parameters_dict = self.priors.sample(size=self.size, returntype='dict')
            dataset_size = self.size
        else:
            if self.parameter_labels != list(parameters_dict.keys()):
                raise ValueError("The prior and the parameter keys do not match")
            dataset_size = np.shape(parameters_dict[list(parameters_dict.keys())[0]])[0]
        # Unless the survey is randomised, we can just generate the grid once
        boxes = []
        surveys = []
        for i in range(self.size):
            box_parameters = dict.fromkeys(self.priors.keys)
            for key in list(self.parameter_labels):
                box_parameters[key] = parameters_dict[key][i]
            box = Box(parameters=box_parameters, density=self.model_framework['density'], background_noise_scale=self.model_framework['noise_scale'])
            noise_prior = Prior(distributions={"noise_scale": self.survey_framework['noise_scale']})
            noise_scale = noise_prior.sample(size=1, returntype='array').flatten()
            survey = GravitySurvey(ranges=self.survey_framework['ranges'], noise_scale=noise_scale, survey_shape=self.survey_framework['survey_shape'], noise_on_location_scale=self.survey_framework['noise_on_location_scale'])
            survey.make_survey()
            # Computing gravity
            if self.model_framework['type'] == 'voxelised':
                box.make_voxel_grid(ranges=self.model_framework['ranges'], grid_shape=self.model_framework['grid_shape'])
            elif self.model_framework['type'] == 'parameterised':
                box.translate_to_parameterised_model()
            else:
                raise ValueError('model_framework type can only be voxelised or parameterised')
            survey_coordinates = survey.survey_coordinates.copy()
            survey.gravity = box.forward_model(survey_coordinates=survey_coordinates, model_type=self.model_framework['type'])
            # Generating random noise
            survey.make_noise()
            boxes.append(box)
            surveys.append(survey)
            if i % 1000 == 0:
                print(f"{i}/{self.size} data points made")
        self.surveys = surveys
        self.boxes = boxes
        return self.surveys, self.boxes

    def make_data_arrays(self, survey_coordinates_to_include=['x', 'y', 'z']):
        """
        Parameters
        ----------
            survey_coordinates_to_include: list
                elements can be 'x' 'y' or 'z', otherwise they are not considered
        """
        # Making the box model array
        if self.model_framework['type'] == 'parameterised':
            data = np.array([self.boxes[i].parameterised_model for i in range(self.size)])
        elif self.model_framework['type'] == 'voxelised':
            data = np.array([self.boxes[i].voxelised_model for i in range(self.size)])
        # Making the survey array
        conditional_gz = np.array([self.surveys[i].gravity for i in range(self.size)])
        noise = np.array([self.surveys[i].noise for i in range(self.size)])
        conditional_gz = conditional_gz + noise # Adding noise
        conditional_coordinates = np.array([self.surveys[i].survey_coordinates for i in range(self.size)])
        conditional = np.expand_dims(conditional_gz, axis=2)
        labels = ['x', 'y', 'z']
        for idx, label in enumerate(labels):
            if any([l==label for l in survey_coordinates_to_include]):
                conditional = np.concatenate((conditional,conditional_coordinates[:,:,idx:idx+1]), axis=2)
        #conditional = conditional.reshape(*conditional.shape[:-2], -1)
        return data, conditional

