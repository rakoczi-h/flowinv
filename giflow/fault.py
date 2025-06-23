import numpy as np
from utils import points_within_area, distance_to_line_segment
from scipy.interpolate import splprep, splev
from scipy.interpolate import LinearNDInterpolator
import math
import plotly.io
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime
from prior import Prior

class Fault:
    def __init__(self, parameters: dict, grid=None, displacement_profile=None, density=800.0):
        self.parameters = parameters
        self.grid = grid
        self.displacement_profile = displacement_profile
        self.density = density
    def __setattr__(self, name, value):
        if name == 'parameters':
            if value is not None:
                if not isinstance(value, dict):
                    raise ValueError("parameters has to be a dictionary")
                default_keys = ["cx", "cy", "alpha", "l", "DL_ratio", "Extent_ratio", "dip", "sym_factor", "Displacement_order", "Blend_order", "depth"]
                for key in value.keys():
                    if not key in default_keys:
                        raise ValueError('At least one of the keys in parameters is not recognised.')
                for df in ['cx', 'cy', 'alpha', 'l']:
                    value.setdefault(df, None)
                value.setdefault("DL_ratio", 0.1)
                value.setdefault("dip", 70/180*np.pi)
                value.setdefault("sym_factor", 0.2)
                value.setdefault("Extent_ratio", 1.5)
                value.setdefault("Displacement_order", 1.2)
                value.setdefault("Blend_order", 1.2)
                value.setdefault("depth", 0.0)
                for i, key in enumerate(default_keys):
                    super().__setattr__(key, value[key])
        super().__setattr__(name, value)

    def make_fault(self, grid=None):
        if grid is None:
            if self.grid is not None:
                grid = self.grid
            else:
                raise ValueError('Need to provide a grid to create the displacement profile')
        else:
            self.grid = grid

        Z = grid[:,:,2].flatten()
        original_shape = np.shape(grid[:,:,2])
        grid = grid[:,:,0:2].reshape(np.shape(grid)[0]*np.shape(grid)[1], 2)

        DL_ratio = self.parameters['DL_ratio']
        dip = self.parameters['dip']
        sym_factor = self.parameters['sym_factor']
        Displacement_Order = self.parameters['Displacement_order']
        Blend_Order = self.parameters['Blend_order']
        l = self.parameters['l']

        if self.parameters['alpha'] < 0.0 or self.parameters['alpha'] > 2*np.pi:
            raise ValueError('The rotation angle should be between 0.0 and 2*pi.')
        elif self.parameters['alpha'] <= np.pi:
            sense = 'rhs'
        elif self.parameters['alpha'] > np.pi:
            sense = 'lhs'

        sides = ['lhs', 'rhs']
        for side in sides:
            if side == 'rhs':
                if sense == 'rhs':
                    max_displacement = l * DL_ratio * sym_factor
                elif sense == 'lhs':
                    max_displacement = -l * DL_ratio
                else:
                    raise ValueError('sense can only be rhs or lhs')
                xyboundary, _ = self.make_rupture_area()
            elif side == 'lhs':
                if sense == 'rhs':
                    max_displacement = -l * DL_ratio
                elif sense == 'lhs':
                    max_displacement = l * DL_ratio * sym_factor
                else:
                    raise ValueError('sense can only be rhs or lhs')
                _, xyboundary= self.make_rupture_area()
            else:
                raise ValueError('side can be rhs or lhs only.')

            if max_displacement == 0:
                print('There is no displacement.')

            # using the polynomial method
            # n = 100
            # xyboundary_coeffs, xyboundary_points = make_angular_bend_curve_v2(xyboundary[:,0], xyboundary[:,1], alpha, n)
            # # points_inside = np.array([[-1, -1.5],[-0.5, -1], [0.5, -1]])
            # dist_a, dist_b = distance_to_curve_v2(points_inside, xyboundary_coeffs)


            # Using the spline method
            n = 100 # must be even
            xyboundary = self.make_angular_bend_curve(xyboundary[:,0], xyboundary[:,1], n)
            displaced_gridpoints = points_within_area(xyboundary, grid)
            
            points_inside = grid[displaced_gridpoints]
            if not points_inside.size:
                self.displacement_profile = self.grid[:,:,2]
                return self.displacement_profile
            # dividing it up into two sides
            xyboundary_a = xyboundary[:int(n/2),:]
            xyboundary_b = xyboundary[int(n/2):,:]
            # points_inside = points_within_area(xyboundary, grid)
            # Calculating the distance from the edge and the corresponding displacement
            dist_a = self.distance_to_curve(xyboundary_a, points_inside)
            dist_b = self.distance_to_curve(xyboundary_b, points_inside)


            distance_from_edge = np.abs((dist_a**(-Displacement_Order) + dist_b**(-Displacement_Order))**(-Displacement_Order))
            d = distance_from_edge / (self.parameters['l']/2) # Normalise. The maximum distance from the edge is the half length of the trace, and the minimum is 0
            taper = ((np.cos(d*np.pi)+1)/2)**(Blend_Order)
            dz1 = (1-taper)*max_displacement

            # Calculating the distance from the trace line - making the fault plane

            x, y = self.translate_coords()
            trace_end_1 = np.array([x[0], y[0]])
            trace_end_2 = np.array([x[1], y[1]])
            distance_to_trace = distance_to_line_segment(points_inside, np.repeat(np.expand_dims(trace_end_1, axis=0), np.shape(points_inside)[0], axis=0), np.repeat(np.expand_dims(trace_end_2, axis=0), np.shape(points_inside)[0], axis=0))

            if max_displacement < 0:
                dz2 = -np.tan(dip)*distance_to_trace
                dz = np.max(np.array([dz1, dz2]), axis=0)
            else:
                dz2 = np.tan(dip)*distance_to_trace
                dz = np.min(np.array([dz1, dz2]), axis=0)

            Z[displaced_gridpoints] = Z[displaced_gridpoints] + dz
        self.displacement_profile = np.reshape(Z, original_shape)
        
        return self.displacement_profile

    # --------------- Functions for making the fault model -----------------
    def make_rupture_area(self):
        """
        Create a polygonal rupture area around a fault trace defined by (x, y).

        Parameters:
            x (array-like): x-coordinates of the fault trace.
            y (array-like): y-coordinates of the fault trace.
            DLRatio (float or list/tuple of two floats): Length-to-width ratio(s) for the rupture area.

        Returns:
            polygon (dict): {'x': ..., 'y': ...} defining the polygon coordinates.
            xyForward (ndarray): Points on the "forward" side of the fault.
            xyBackward (ndarray): Points on the "backward" side of the fault.
        """
        x, y = self.translate_coords()
        Extent_Ratio = self.parameters['Extent_ratio']

        # Two different values can be given for the backwards and forwards part of the fault
        # if one is given, then it is reused for both sides
        if np.isscalar(Extent_Ratio):
            Extent_Ratio = [Extent_Ratio, Extent_Ratio]
        elif len(Extent_Ratio) == 1:
            Extent_Ratio = [Extent_Ratio[0], Extent_Ratio[0]]
        elif len(Extent_Ratio) > 2:
            raise ValueError('The maximum number of elements in Extent_Ratio is 2.')

        # Compute cumulative distances
        dx = x[1]-x[0]
        dy = y[1]-y[0]
        trace_length = np.sqrt(dx**2 + dy**2)

        xyForward = [[x[0], y[0]]]
        xyBackward = [[x[0], y[0]]]

        xMid = (x[0] + x[1]) / 2
        yMid = (y[0] + y[1]) / 2
        dist = trace_length/2

        widthF = dist * Extent_Ratio[0]
        widthB = dist * Extent_Ratio[1]

        if (y[1]-yMid) == 0: # need to deal with this separately, as otherwise dividing by 0
            alpha = np.pi/2
        else:
            alpha = np.arctan((x[1]-xMid)/(y[1]-yMid))

        xp1 = xMid + widthF*np.cos(alpha)
        yp1 = yMid - widthF*np.sin(alpha)
        xyForward.append([xp1, yp1])

        xp2 = xMid - widthB*np.cos(alpha)
        yp2 = yMid + widthB*np.sin(alpha)
        xyBackward.append([xp2, yp2])

        # Append the final point
        xyForward.append([x[-1], y[-1]])
        xyBackward.append([x[-1], y[-1]])

        xyForward = np.array(xyForward)
        xyBackward = np.array(xyBackward)

        return xyForward, xyBackward
    
    def make_angular_bend_curve(self, xp, yp, n):
        """
        Generates a smooth closed curve from points (xp, yp) using parametric spline interpolation.

        Parameters:
            xp (array-like): x-coordinates of input points.
            yp (array-like): y-coordinates of input points.
            n (int): Number of points to interpolate on the curve.

        Returns:
            np.ndarray: An (n, 2) array with interpolated x and y coordinates.
        """
        if not isinstance(xp, np.ndarray) or not isinstance(yp, np.ndarray):
            raise ValueError('xp and yp have to be arrays')
        if xp.ndim != 1 or yp.ndim !=1:
            raise ValueError('xp and yp have to be vectors.')

        # Define angular parameter over [0, 2π]
        th = np.linspace(0, 2 * np.pi, len(xp))

        # Create a periodic spline parameterization
        tck, _ = splprep([xp, yp], u=th, k=2) # the degree is set to 2
        # Evaluate the spline at n points
        th_new = np.linspace(0, 2 * np.pi, n)
        x_new, y_new = splev(th_new, tck)

        # Return as (n, 2) array
        xy_curve = np.vstack((x_new, y_new)).T
        return xy_curve
    
    def translate_coords(self):
        """
        Defines translation between end points x1, y1, x2, y2 and central location, lenght and rotation angle (cx, cy, l, alpha)
        """
        cx = self.parameters['cx']
        cy = self.parameters['cy']
        alpha= self.parameters['alpha']
        l = self.parameters['l']
        x1 = l/2*np.cos(alpha)+cx
        y1 = l/2*np.sin(alpha)+cy
        x2 = -l/2*np.cos(alpha)+cx
        y2 = -l/2*np.sin(alpha)+cy
        return [x1, x2], [y1, y2]
    
    def distance_to_curve(self, xyboundary, points):
        # Getting the arrays to the right shape for parallelization
        x = points[:,0]
        x = np.repeat(np.expand_dims(x, axis=0), np.shape(xyboundary)[0], axis=0)
        y = points[:,1]
        y = np.repeat(np.expand_dims(y, axis=0), np.shape(xyboundary)[0], axis=0)
        xyboundary = np.expand_dims(xyboundary, axis=2)
        xyboundary = np.repeat(xyboundary, np.shape(points)[0], axis=2)
        # finding the minimum distance for each grid point and the point on the curve it corresponds to
        dist = np.sqrt((xyboundary[:,0,:]-x)**2 + (xyboundary[:,1,:]-y)**2)
        min_loc = np.argmin(dist, axis=0)
        min_dist = np.min(dist, axis=0)
        min_points = xyboundary[min_loc, :, 0] # the minimum distance from the original set of points on the boundary

        # Finding the neighbouring points to the original points corresponding to the minimal distances
        min_points_reshaped = np.repeat(np.expand_dims(min_points, axis=0), np.shape(xyboundary)[0], axis=0)
        dist = np.sqrt((xyboundary[:,0,:]-min_points_reshaped[:,:,0])**2 + (xyboundary[:,1,:]-min_points_reshaped[:,:,1])**2)
        dist_ordered = np.argsort(dist, axis=0)
        min_points1 = xyboundary[dist_ordered[1], :, 0]
        min_points2 = xyboundary[dist_ordered[2], :, 0]
        # drawing lines from the original point to the two neighbouring points and finding the minimum distance to those lines
        min_dist1 = distance_to_line_segment(points, min_points, min_points1)
        min_dist2 = distance_to_line_segment(points, min_points, min_points2)
        # picking the minimum value from the final three obtained values.
        min_dist = np.min(np.vstack([min_dist, min_dist1, min_dist2]), axis=0)
        return min_dist
    
    # --------------- Functions to compute survey ----------------------

    def forward_model(self, num_components=50, depth=None, survey_coordinates=None, remove_min=True):
        """
        The fourier domain forward model. based on R.L Parker (1972)
        Takes the displacement model, depth and density contrast and turns it into a gravity signal on the surface. The surface is assume to be completely flat.
        Parameters
        ----------
            num_compontnents: int
                The number of fourier components to consider
            depth: float
                depth of the fault. If not give, try to use the depth from the parameter dictionary. The units are km.
            survey_coordinates: array
                [survey_points, 3] the survey coordiants in 3 dimensions, in the order of x, y z
                if not given, then the same grid is used as for the fault.
        """
        if self.displacement_profile is None:
            self.make_fault()
        if depth is None:
            depth = self.parameters["depth"]
            if depth is None:
                raise ValueError("Need to provide the depth of the fault")
        k_mag = self.make_k_vector()
        if np.isnan(self.displacement_profile).any():
            print('Found NaN in displacement model')
        # eq.5 in R. L. Parker (1972)
        N = num_components
        R1 = np.zeros(np.shape(self.displacement_profile))
        for n in range(N):
            f1 = np.fft.fft2((self.displacement_profile*1000.0)**(n+1)) # changing to m
            r1 = np.complex128(k_mag**(n)/math.factorial(n+1)*f1) # r represent spatial domain, k represent k domain
            R1 = R1+r1
        self.fourier_domain_model = R1
        G = 6.67430*10**(-11) # Nm**2kg**(-2)
        f_g = -2*np.pi*G*np.exp((-k_mag)*depth*1000.0)*R1*self.density
        g = np.fft.ifft2(f_g)
        g_vec = g.ravel()
        g_orig = np.real(g_vec) * 1e5 # changing to mGal
        if survey_coordinates is None:
            if remove_min:
                return g_orig-np.min(g_orig), R1
            else:
                return g_orig, R1
        else:
            coords = np.reshape(self.grid[:,:,:2], (np.shape(self.grid)[0]*np.shape(self.grid)[1], 2))
            func = LinearNDInterpolator(coords, g_orig)
            g_new = func(survey_coordinates[:,0], survey_coordinates[:,1])
            if remove_min:
                return g_new-np.min(g_new), R1
            else:
                return g_new, R1
            
    def make_k_vector(self):
        X = self.grid[:,:,0]*1000.0 # changing to m
        Y = self.grid[:,:,1]*1000.0
        numrows = np.shape(X)[0]
        numcolumns = np.shape(Y)[0]
        longx = np.max(X)-np.min(X)
        longy = np.max(Y)-np.min(Y)
        frequency = np.zeros((abs((numrows // 2) + 1), abs((numcolumns // 2) + 1)))

        for f in range(1, int((numrows/2) + 2)):
            for g in range(1, int((numcolumns/2) + 2)):
                frequency[f-1, g-1] = np.sqrt(((f-1) / longx) ** 2 + ((g-1) / longy) ** 2)

        frequency2 = np.fliplr(frequency)
        frequency3 = np.flipud(frequency)
        frequency4 = np.fliplr(np.flipud(frequency))

        entero = round(numcolumns / 2)
        if ((numcolumns / 2) - entero) == 0:
            frequency2 = np.delete(frequency2, 0, axis=1)
            frequency3 = np.delete(frequency3, 0, axis=0)
            frequency4 = np.delete(np.delete(frequency4, 0, axis=1), 0, axis=0)
            frequencytotal = np.concatenate((np.concatenate((frequency, frequency2), axis=1),
                                            np.concatenate((frequency3, frequency4), axis=1)))
        else:
            frequencytotal = np.concatenate((np.concatenate((frequency, frequency2), axis=1),
                                            np.concatenate((frequency3, frequency4), axis=1)))
        frequencytotal = frequencytotal[:-1, :]
        frequencytotal = frequencytotal[:, :-1]
        frequencytotal = frequencytotal * (2 * np.pi)
        return frequencytotal
    

    # ------------------- Plotting Tools ---------------------------

    def plot_pixels(self, filename='', survey_coordinates=None):
        """
        Creates a simple pixelised image of the survey. Can only be done for gridded data.
        """
        if self.displacement_profile is None:
            raise ValueError("displacement profile not given")

        plt.imshow(np.reshape(self.displacement_profile, np.shape(self.grid[:,:,0])), extent=(np.min(self.grid[:,:,0]), np.max(self.grid[:,:,0]), np.max(self.grid[:,:,1]), np.min(self.grid[:,:,1])))
        plt.colorbar(label='km')
        if survey_coordinates is not None:
            plt.scatter(survey_coordinates[:,0], survey_coordinates[:,1], s=1, marker='o', color='black')
        plt.scatter(self.parameters['cx'], self.parameters['cy'], color='red', marker='x', s=2)
        plt.xlabel('x [km]')
        plt.ylabel('y [km]')
        plt.savefig(filename)
        plt.close()

    def plot_3D_surface(self, survey_coordinates=None, filename='', depth=None):
        """
        Makes 3D plot of the fault.
        Parameters
        ----------
            filename : str
                The location where the image will be saved. The file format is infered from this string, can either be .png or .html
            depth: float
                The depth of the fault. If not given, try to get from parameters dictionary or set to 0.
        """
        plotly.io.templates.default = 'plotly_white'
        if self.displacement_profile is None:
            raise ValueError("displacement profile not given")
        X = self.grid[:,:,0]
        Y = self.grid[:,:,1] # changing to km
        model = self.displacement_profile/1000.0
        depth = self.parameters['depth']
        model = np.reshape(model, np.shape(X))
        aspect_ratios = [1, 1, 0.2]
        fig = go.Figure(data=[go.Surface(z=model-depth, x=X, y=Y, cmax=np.max(-depth)-depth/100, cmin=np.min(-depth)+depth/100,
                                      contours = {"z": {"show": True, "start": np.min(model-depth), "end": np.max(model-depth), "size": (np.max(model-depth)-np.min(model-depth))/10, "color":"black"}},
                                      colorbar={"title": 'z [km]'})])

        if not survey_coordinates is None:
            fig.add_scatter3d(x=survey_coordinates[:,0], y=survey_coordinates[:,1], z=survey_coordinates[:,2], mode='markers', marker={'color': 'black', 'size': 1, 'opacity': 0.2}, showlegend=None)
            fig.update_layout(coloraxis_showscale=False)

        fig.update_scenes(aspectratio={"x":aspect_ratios[0], "y":aspect_ratios[1], "z":aspect_ratios[2]},
                          zaxis={"nticks": 2, "range": [np.min(model-depth), 0.01]},
                          xaxis={'range': [np.min(X), np.max(X)]},
                          yaxis={'range': [np.min(Y), np.max(Y)]},
                          xaxis_title="x [km]",
                          yaxis_title="y [km]",
                          zaxis_title="z [km]"
        )
        camera = dict(
            eye=dict(x=0.8, y=0.8, z=0.6))
        fig.update_layout(scene_camera=camera)
        if filename[-5:] == '.html':
            fig.write_html(filename)
        elif filename[-4:] == '.png':
            fig.write_image(filename)
        else:
            raise ValueError("Only .html and .png file extensions are allowed")
        plt.close()

class FaultDataset():
    """
    Class for making a data set of faults and corresponding gravity surveys.
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
        faults: list
            the list of the generated fault objects
        surveys: list
             the list of the generated surveys
    """
    def __init__(self, size: int, priors: Prior, model_framework={}, survey_framework={}):
        self.size = size
        self.priors = priors
        self.model_framework = model_framework
        self.survey_framework = survey_framework
        self.faults = None
        self.surveys = None

    def __setattr__(self, name, value):
        if name == 'model_framework':
            if not isinstance(value, dict):
                raise ValueError("Expected dict for model_framework.")
            value.setdefault("parameters_to_include", ['cx', 'cy', 'l', 'alpha'])
            value.setdefault("density", 800.0)
            value.setdefault("grid_ranges", None)
            value.setdefault("grid_resolution", None)
        if name == 'survey_framework':
            if not isinstance(value, dict):
                raise ValueError("Expected dict for survey_framework.")
            value.setdefault("noise_scale", 0.0)
            value.setdefault("sizes", [1,1,0])
            value.setdefault("survey_shape", [5,5])
            if not isinstance(value["survey_shape"], list):
                raise ValueError("The survey shape has to be a list. Can have a single element")
        super().__setattr__(name, value)

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
        X = np.linspace(-2, 2, num=50)
        Y = np.linspace(-2, 2, num=50)
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

