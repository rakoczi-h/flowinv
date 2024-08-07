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
            The keys that are understood by the functions within this class are: px, py, pz (the coordinates of the centre of the box, in meters), lx, ly, lz (the size of the box in the different directions, in meters), alpha (the rotations angle of the box around the z axis, in radians) (Default: None)
        parameter_labels: list
            Defaults to the keys in the parameters dictionary. (Default: None)
        density: float
            The density contrast between the box and the background in kg/m^3 (Default: None)
        parameterised_model: np.array
            The parameterised model for the box. This is only used when turning the box into a member of a data set used for training a neural network. (Default: None)
        voxelised_model: np.array
            The voxelised model of the box. An array of density values. (Default: None)
        voxel_grid: np.array
            The grid that the densities are defined on. (Default: None)
        background_noise_scale: float
            The scale of the gaussian noise added to simulate background effects. This is only used when a voxelised representation is generated of the box. (Default: 0.0)
    """
    def __init__(self, parameters=None, density=None, parameterised_model=None, voxelised_model=None, background_noise_scale=0.0, parameter_labels=None):
        self.density = density
        self.voxelised_model = voxelised_model
        self.parameterised_model = parameterised_model
        self.background_noise_scale=background_noise_scale
        self.parameter_labels = parameter_labels
        self.parameters = parameters 
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
            survey_coordinates: np.array
                has to contain the coordinates of the x y z survey locations [number of survey points, 3]
            model_type: str
                Define which model type to compute the forward model from. Can be parameterised or voxelised. (Default: 'parameterised')
        Output
        ------
            gz: 1D np.array
                The gravity anomaly corresponding to the box. The size is the number of survey coordinates. 
        """
        if model_type=='parameterised':
            if self.parameters is None:
                raise ValueError('The parameters are not defined. Either define the parameters of set from_voxelised to True.')

            survey_coordinates[:,:2] = rotate(survey_coordinates[:,:2], origin=(self.px, self.py), angle=-self.alpha)
            limits = np.expand_dims(np.array([[self.px-self.lx/2,self.px+self.lx/2], [self.py-self.ly/2,self.py+self.ly/2], [self.pz-self.lz,self.pz]]), axis=0)
            gz = self.get_gz(limits=limits, rho=self.density, survey_coordinates=survey_coordinates)
        elif model_type=='voxelised':
            if self.voxel_grid is None:
                raise ValueError("The voxel_grid is not defined")
            if self.voxelised_model is None:
                self.translate_to_voxels()
            gz = self.get_gz(limits=self.voxel_grid, rho=self.voxelised_model, survey_coordinates=survey_coordinates)
        else:
            raise ValueError('model_type can only be parameterised or voxelised.')
        gz = gz-np.min(gz)
        return gz


    def get_gz(self, survey_coordinates, limits, rho):
        """
        Analytical forward model based on Jung (1961) and Plouff (1966), review Li (1998). Calculated the vertical gravity at a set of measurement locations from a number of rectangular prisms.
        Parameters
        ----------
            survey_coordinates: array
                [number of measurement locations, (x,y,z)] The locations where the gravity measurements are taken.
            limits: array
                [number of prisms, (x,y,z), (min, max)] The array defining the limits of the prisms that we want to see the gravity signature of. This would be 1 lenght, if we are looking for the gravity from a single box based on its parameters. In the voxelised case, the length of this is the number of voxels in the model.
            rho: float or array
                If it is a float, then all the voxels have this same density. If an array, it has to have the same lenght as the first dimension of limits.
        Output
        ------
            Returns the vector of vertical gravity values at the survey_coordinates locations. (microGal)
        """

        shape = (np.shape(limits)[0], np.shape(survey_coordinates)[0])

        x = survey_coordinates[:,0]
        y = survey_coordinates[:,1]
        z = survey_coordinates[:,2]

        xi1 = limits[:,0,0]
        xi2 = limits[:,0,1]
        eta1 = limits[:,1,0]
        eta2 = limits[:,1,1]
        zeta1 = limits[:,2,0]
        zeta2 = limits[:,2,1]

        G = 6.67408E-11
        # Vectorize
        x = np.broadcast_to(x, shape)
        y = np.broadcast_to(y, shape)
        xi1 = np.broadcast_to(xi1[:, np.newaxis], shape)
        xi2 = np.broadcast_to(xi2[:, np.newaxis], shape)
        eta1 = np.broadcast_to(eta1[:, np.newaxis], shape)
        eta2 = np.broadcast_to(eta2[:, np.newaxis], shape)
        zeta1 = np.broadcast_to(zeta1[:, np.newaxis], shape)
        zeta2 = np.broadcast_to(zeta2[:, np.newaxis], shape)
        if isinstance(rho, np.ndarray):
            rho = np.broadcast_to(rho[:, np.newaxis], shape)


        x1 = x - xi1
        x2 = x - xi2
        y1 = y - eta1
        y2 = y - eta2
        z1 = z - zeta1
        z2 = z - zeta2

        r111 = np.sqrt(x1 * x1 + y1 * y1 + z1 * z1)
        r211 = np.sqrt(x2 * x2 + y1 * y1 + z1 * z1)
        r121 = np.sqrt(x1 * x1 + y2 * y2 + z1 * z1)
        r112 = np.sqrt(x1 * x1 + y1 * y1 + z2 * z2)
        r212 = np.sqrt(x2 * x2 + y1 * y1 + z2 * z2)
        r221 = np.sqrt(x2 * x2 + y2 * y2 + z1 * z1)
        r122 = np.sqrt(x1 * x1 + y2 * y2 + z2 * z2)
        r222 = np.sqrt(x2 * x2 + y2 * y2 + z2 * z2)

        u111 = -1
        u211 = 1
        u121 = 1
        u112 = 1
        u212 = -1
        u221 = -1
        u122 = -1
        u222 = 1

        t111 = u111 * (x1 * np.log(y1 + r111) + y1 * np.log(x1 + r111) - z1 * np.arctan((x1 * y1) / (z1 * r111)))
        t211 = u211 * (x2 * np.log(y1 + r211) + y1 * np.log(x2 + r211) - z1 * np.arctan((x2 * y1) / (z1 * r211)))
        t121 = u121 * (x1 * np.log(y2 + r121) + y2 * np.log(x1 + r121) - z1 * np.arctan((x1 * y2) / (z1 * r121)))
        t112 = u112 * (x1 * np.log(y1 + r112) + y1 * np.log(x1 + r112) - z2 * np.arctan((x1 * y1) / (z2 * r112)))
        t212 = u212 * (x2 * np.log(y1 + r212) + y1 * np.log(x2 + r212) - z2 * np.arctan((x2 * y1) / (z2 * r212)))
        t221 = u221 * (x2 * np.log(y2 + r221) + y2 * np.log(x2 + r221) - z1 * np.arctan((x2 * y2) / (z1 * r221)))
        t122 = u122 * (x1 * np.log(y2 + r122) + y2 * np.log(x1 + r122) - z2 * np.arctan((x1 * y2) / (z2 * r122)))
        t222 = u222 * (x2 * np.log(y2 + r222) + y2 * np.log(x2 + r222) - z2 * np.arctan((x2 * y2) / (z2 * r222)))

        # Sum contributions over the voxel dimensions
        return 1E8*-G*np.sum(rho * (t111 + t211 + t121 + t112 + t212 + t221 + t122 + t222), axis=0)


    def translate_to_voxels(self, voxel_grid=None, background_noise_scale=None, density=None):
        """Makes a rotated box with the given parameters [px, py, pz, lx, ly, lz, alpha translated
        to density values at the voxel grid location specified.
        The baseline background density is assumed to be 0 and the density of the object is defined relative to that.
        Parameters
        ----------
            voxel_grid: array
                Needs to contain the boundaries of the voxels. [number of voxels, 3, 2]
                Coordinate order is xyz given in the 2nd dimension. Limits are 2, min and max, given in the 3rd dimension. (Default: None)
            density: float
                The density of the box object, relative to 0 background. (kg/m^3) (Default: None)
            background_noise_scale: float
                The standard deviation of the Gaussian background noise that is added to the model. If None, the value stored in the class is used. (Default: None)
                CAUTION: overwrites previous value if given.
        Output
        ------
            voxelised_model: array
                The voxelised representation of the box defined on the input grid and parameters. [number of voxels] Consists of density values.
            voxel_grid: array
                The grid the density values are defined on. 
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
        rho_bg = 0.0 # bg assumed to be 0 density
        bg = np.zeros(num_voxels)
        if self.background_noise_scale != 0.0: # adding noise
            bg = bg + np.random.normal(loc=0.0, scale=self.background_noise_scale, size=np.shape(bg)[0])

        # calculating the locations of the vertices of the rectangle, which is the box when viwed from above
        p1 = tuple(rotate(np.array([self.px-self.lx/2, self.py+self.ly/2]), origin=(self.px,self.py), angle=self.alpha))
        p2 = tuple(rotate(np.array([self.px+self.lx/2, self.py+self.ly/2]), origin=(self.px,self.py), angle=self.alpha))
        p3 = tuple(rotate(np.array([self.px-self.lx/2, self.py-self.ly/2]), origin=(self.px,self.py), angle=self.alpha))
        p4 = tuple(rotate(np.array([self.px+self.lx/2, self.py-self.ly/2]), origin=(self.px,self.py), angle=self.alpha))
        # making the 1d rectangle
        rect1 = Polygon([p1, p2, p4, p3])
        # defining the extent of the box in the z direction
        z_top = self.pz
        z_bottom = self.pz-self.lz
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
        """
        Translates the parameters to a parameterised model that can be used for training.
        Parameters
        ----------
            keys: list
                The parameter labels to use in the parameterised model. Useful to give when eg. we want to marginalise over some box parameters. If not given, parameter_labels is used. (Default: None)
        Output
        ------
            parameterised_model: np.array
                The array of parameters of the box.
        """
        if keys is None:
            keys = self.parameter_labels
        self.parameterised_model = np.array([self.parameters[key] for key in keys])
        return self.parameterised_model

    def translate_to_parameters(self):
        """
        Takes the parameterised_model and the parameter_labels and translated it into a dictionary.
        Output
        ------
            parameters: dict
                Dictionary of the parameters of the box.
        """
        if self.parameterised_model is None:
            raise ValueError("Need to define parameterised model")
        if self.parameter_labels is None:
            raise ValueError("Need to define parameter labels")
        parameters = dict.fromkeys(self.parameter_labels)
        for idx, key in enumerate(self.parameter_labels):
            parameters[key] = self.parameterised_model[idx]
        self.parameters = parameters
        return self.parameters

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
            voxel_grid: array
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
            filename: str
                Where to save plot. (Default: 'voxel_slices.png')
            slices_per_dimension: int
                The number of slices to pick per dimension,
                Cannot be more than the number of voxels in that dimension. (Default: 3)
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

    def plot_3D_mesh(self, axis_limits=None, filename='3D_mesh.html'):
        """
        Makes a 3D image of the box from its parameters.
        Parameters:
            axis_limits: array
                np.array([[xmin, xmax], [ymin,ymax], [zmin,zmax]])
                if None, it is infered from the voxel_grid or is set to a default value. (Default: None)
            filename: str
                Where to save plot. Extension can be .html (interactive plot) of .png. (Default: '3D_mesh.html')
        """
        # define the ranges of the coordinates
        x1 = self.px - self.lx/2
        x2 = self.px + self.lx/2
        y1 = self.py - self.ly/2
        y2 = self.py + self.ly/2
        z1 = self.pz - self.lz
        z2 = self.pz
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
                zmin = np.min(np.mean(self.voxel_grid[:,2,:], axis=1))
                zmax = np.max(np.mean(self.voxel_grid[:,2,:], axis=1))
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

    def plot_3D_volume(self, filename='3D_volume.html'):
        """
        Creates a 3D plot from the voxelised representation of the box.
        Parameters:
        ----------
            filename: str
                Where to save plot. Extension can be .html (interactive plot) of .png. (Default: '3D_volume.html')
        """

        x = np.mean(self.voxel_grid[:,0,:], axis=1)
        y = np.mean(self.voxel_grid[:,1,:], axis=1)
        z = np.mean(self.voxel_grid[:,2,:], axis=1)

        fig = go.Figure(data=go.Volume(x=x, y=y, z=z, value=self.voxelised_model,
                    opacity = 0.4,
                    surface_count = 17))
        fig.update_layout(height = 800,
                          width = 1000,
                          font = dict(size=12),
                          coloraxis_colorbar={"title": r'$\rho [kg/m^3$]'},
                          scene = dict(
                               xaxis_title='x [m]',
                               yaxis_title='y [m]',
                               zaxis_title='z [m]',
                               aspectmode='manual'))

        if filename[-5:] == '.html':
            fig.write_html(filename)
        elif filename[-4:] == '.png':
            fig.write_image(filename)
        else:
            raise ValueError("Only .html and .png file extensions are allowed")
        plt.close()

class BoxDataset:
    """
    Class for making and handling a data set consisting of box objects and the corresponding gravimetry surveys.
    Parameters
    ----------
        size: int
            The number of data points in the data set
        priors: Prior
            Defines the prior distributions from which the box parameters are contained within.
        model_framework: dict
            Defines the parameters to set up the model configuration.
                'type': Can be 'voxelised' or 'parameterised' (Default: 'parameterised')
                'density': The relative density of the box compared to background (Default: -2670.0 kg/m^3)
                'noise_scale': Defines the std of the Gaussian noise added to the background of the voxelised model, ignored when the model is parametersied. Can be an int or a list. If int, the noise scale is set to this for each member of the data set. If list, it has to be of the form ['Uniform', low, high] or ['Normal', loc, scale]. (Default: 0.0)
                'grid_shape': List with length 3, each member defining the number of voxels in the 3 dimensions. Eg. [10,10,10]. Only used if 'type'=='voxelised'. (Default: None)
                'ranges': Nested list that defines the range in space that is covered by the voxelised model in the 3 dimensions. [[xmin, xmax],[ymin, ymax],[zmin, zmax]] (Default: None)
        survey_framework: dict
            Defines the parameters that set up the configuration of the simualted surveys.
                'noise_scale': Defines the std of the Gaussian noise that simulates the noise on the gravity survey. Defined the same as for model_framework. (Default: 0.0)
                'survey_shape': Defines the number of measurement locations in the various dimensions. List or int. If int, the survey locations are randomly drawn within the are. If list with 2 elements, the z value is kept constant and a grid with the defined shape is generated. If list with 3 elements, then the z value is gridded too.
                'ranges': Defines the area covered by the survey. Defined the same way as for model_framework. If z is set to be at 0, we can define this as eg. [[-1,1], [1,1], [0]]. (Default: [[-0.5,0.5], [-0.5,0.5], [0]]
                'noise_on_location_scale': The Gaussian noise added to the geenrated locations of the survey points. (Default: 0.0)
        boxes: list
            A list of Box objects.
        surveys: list
            A list of survey objects.
    """
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
            value.setdefault("density", -2670.0)
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
                Otherwise the parameters are taken from the dictionary (Default: None)
        Output
        ------
            boxes: list of Box objects
            surveys: list of Survey objects
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
        i = 0
        while i < self.size:
            box_parameters = dict.fromkeys(self.priors.keys)
            for key in list(self.parameter_labels):
                box_parameters[key] = parameters_dict[key][i]

            # Making the box object
            box = Box(parameters=box_parameters, density=self.model_framework['density'])
            # Making the survey object
            survey = GravitySurvey(ranges=self.survey_framework['ranges'], survey_shape=self.survey_framework['survey_shape'], noise_on_location_scale=self.survey_framework['noise_on_location_scale'])
            survey.make_survey() # making the survey grid
            # Computing gravity
            if self.model_framework['type'] == 'voxelised':
                # Generating random background noise for the voxelised representation
                bg_noise_prior = Prior(distributions={"bg_noise_scale": self.model_framework['noise_scale']})
                bg_noise_scale = bg_noise_prior.sample(size=1, returntype='dict')['bg_noise_scale'][0]
                box.background_noise_scale = bg_noise_scale
                box.make_voxel_grid(ranges=self.model_framework['ranges'], grid_shape=self.model_framework['grid_shape'])
            elif self.model_framework['type'] == 'parameterised':
                box.translate_to_parameterised_model()
            else:
                raise ValueError('model_framework type can only be voxelised or parameterised')
            survey_coordinates = survey.survey_coordinates.copy()
            survey.gravity = box.forward_model(survey_coordinates=survey_coordinates, model_type=self.model_framework['type']) # assigning the simualted gravity to the survey

            # Checking for inf and nan
            if np.isinf(np.max(survey.gravity)):
                print("Found inf gravity")
                continue
            if np.isnan(np.min(survey.gravity)):
                print("Found nan gravity")
                continue

            # Generating random noise for the survey
            noise_prior = Prior(distributions={"noise_scale": self.survey_framework['noise_scale']})
            noise_scale = noise_prior.sample(size=1, returntype='dict')['noise_scale'][0]
            survey.noise_scale = noise_scale
            survey.make_noise()
            # Adding to the list
            boxes.append(box)
            surveys.append(survey)
            if i % 1000 == 0:
                print(f"{i}/{self.size} data points made")
            i = i+1
        self.surveys = surveys
        self.boxes = boxes
        return self.surveys, self.boxes

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

        # Making the box model array
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

