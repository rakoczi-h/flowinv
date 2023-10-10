import numpy as np
import json
import h5py
from datetime import datetime
from random import random
from shapely.geometry import Polygon
import os

from utils import rotate
from forward_models import get_gz_analytical_vectorised, get_gz_analytical

np.random.seed(1234)

# --------------------------------------------- Data set generation ---------------------------------------------
def make_parameterised_box_dataset(survey_coords, noise_on_grid=False, device_noise=False, testdata=False, filename='analytical_box_dataset.hdf5', dataloc='', datasize=100, saving=True):
    """Reads in box parameters from a file (optional, if file exists) and analytically calculates the gravitational signature of those boxed at the specified survey locations.
    The px, py, pz coordinates of the box are used to shift the survey points, the alpha coordinates are used to rotationally translate them.
    The dimensions of the box are then used in the analytical expression.
    Noise can be addedd optionally to the measurement locations and the gravity values themselves.
    A params_makedata.json file needs to exist for this method to work. This is where the data saving location, data size, and noise levels are specified.
    Parameters
    ----------
        survey_coords: array
            Contains the survey location xyz coordinates. [no. of survey locations, 3]
        noise_on_grid: bool
            If True, Gaussian noise is added to the survey locations, with std specified in dataset_params.json
        device_noise: bool
            If True Gaussian noise is added to the gravity values, with std specified in dataset_params.json
        testdata: bool
            If True then the box_params_test.hdf5 is searched for.
            If False then the box_params_train.hdf5 is searched for.
            If netiher are found, then new parameters are generated with datasize given as an input to this method
        filename: str,
            The name under which the generated dataset will be saved
        dataloc: str,
            The location where to save the data. (if saving)
        datasize: int,
            The size of the dataset to be made. (ignored if data is made from pre-made parameters file)
        saving: bool, if True, the dataset is saved, is not, the generated data is only returned as the output
    Output
    ------
        models: array
            The generated set of parameters. [no. of datapoints, no. of parameters]
        surveys: array
            The forward modelled surveys corresponding to each set of parameters. [np. of datapoints, no. of survey locations]
    """
    if saving:
        path_to_datafile = os.path.join(dataloc, filename)
        if os.path.exists(path_to_datafile):
            raise FileExistsError("File with this name already exists.")
    if not os.path.exists('dataset_params.json'):
        raise FileNotFoundError('The dataset_params.json file has to exist.')
    with open('dataset_params.json') as json_file:
        params = json.load(json_file)
    # Getting the box parameters
    if testdata == True:
        path_to_paramsfile = os.path.join(params['dataloc'], 'box_params_test.hdf5')
    else:
        path_to_paramsfile = os.path.join(params['dataloc'], 'box_params_train.hdf5')
    if os.path.exists(path_to_paramsfile):
        print("Making data from parameters file.")
        h = h5py.File(path_to_paramsfile, "r")
        px = np.array(h['px'])
        py = np.array(h['py'])
        pz = np.array(h['pz'])
        lx = np.array(h['lx'])
        ly = np.array(h['ly'])
        lz = np.array(h['lz'])
        alpha_x = np.array(h['alpha_x'])
        alpha_y = np.array(h['alpha_y'])
        h.close()
        datasize = np.shape(px)[0] # if we are using the new dataset, the data size is inferred from this data set.
        print(f"Data set size inferred from file: {datasize}. Given value overwritten.")
    else: # if these have not been pre-made, make them now
        print("Data set randomly generated.")
        n_per_side = params['n_per_side']
        widths_grid = params['widths_grid']
        voxel_grid, widths_voxels = make_3D_grid(widths_grid, [n_per_side, n_per_side, n_per_side])
        px, py, pz, lx, ly, lz, alpha_x, alpha_y = make_box_params(voxel_grid, widths_voxels, datasize, saving=False)
        print(f"Data set size given: {datasize}")
    alpha = np.arctan(alpha_y/alpha_x)
    rho = params['delta_rho']
    surveys = []
    models = []
    survey_trans = np.zeros(np.shape(survey_coords))
    starttime = datetime.now()
    for i in range(datasize): # looping over each instance of set of coordinates and computing the survey for each
        survey_c = np.zeros(np.shape(survey_coords))
        survey_c[:,0] = survey_coords[:,0]
        survey_c[:,1] = survey_coords[:,1]
        survey_c[:,2] = survey_coords[:,2]
        if noise_on_grid == True:
            if i == 0:
                print(f"Noisy grid. Survey grid noise scale = {params['surveygrid_noise_scale']}")
            survey_c[:,0] = survey_coords[:,0] + np.random.normal(loc=0.0, scale=params['surveygrid_noise_scale'], size=np.shape(survey_c[:,0])[0])
            survey_c[:,1] = survey_coords[:,1] + np.random.normal(loc=0.0, scale=params['surveygrid_noise_scale'], size=np.shape(survey_c[:,1])[0])
            survey_c[:,2] = survey_coords[:,2]
        # transforming the survey coordinates
        survey_trans[:,:2] = rotate(survey_c[:,:2], origin=(px[i],py[i]), angle=alpha[i])
        survey_trans[:, 2] = survey_c[:,2]
        limits = np.array([[px[i]-lx[i]/2,px[i]+lx[i]/2], [py[i]-ly[i]/2,py[i]+ly[i]/2], [pz[i]-lz[i]/2,pz[i]+lz[i]/2]])
        # calculating the survey
        gz = get_gz_analytical(limits, rho, survey_trans)
        if device_noise == True:
            if i == 0:
                print(f"Noisy survey. Device noise scale = {params['device_noise_scale']}")
            gz = gz + np.random.normal(loc=0.0, scale=params['device_noise_scale'], size=np.shape(gz)[0])
        parameters = np.array([px[i], py[i], pz[i], lx[i], ly[i], lz[i], alpha_x[i], alpha_y[i]])
        surveys.append(np.concatenate((np.expand_dims(gz, axis=1), survey_c), axis=1))
        models.append(parameters)
        if i % 1000 == 0:
            nowtime = datetime.now()
            print(f"Data generated {i}/{datasize} \t Time elapsed: {nowtime-starttime}")
    surveys = np.array(surveys)
    models = np.array(models)
    if saving==True:
        h = h5py.File(path_to_datafile, "a")
        h.create_dataset("models_parameterised", data=models)
        h.create_dataset("surveys", data=surveys)
        h.create_dataset("noise_on_grid", data=noise_on_grid)
        if noise_on_grid == True:
            h.create_dataset("surveygrid_noise_scale", data=params['surveygrid_noise_scale'])
        h.create_dataset("device_noise", data=device_noise)
        if device_noise == True:
            h.create_dataset("device_noise_scale", data=params['device_noise_scale'])
        h.close()
    return models, surveys

def make_voxelised_box_dataset(voxel_coords, survey_coords, noise_on_grid=False, device_noise=False, background_noise=True, testdata=False,
                               filename='analytical_box_voxelised_dataset.hdf5', dataloc='', datasize=100, saving=True):
    """ Reads in box parameters from a file (optional, if file exists), translates these into a voxelised model representation and analytically calculates
    the gravitational signature of those boxed at the specified survey locations.
    Noise can be addedd optionally to the measurement locations and the gravity values themselves, as well as density noise to the background of the box.
    A params_makedata.json file needs to exist for this method to work. This is where the data saving location, data size, and noise levels are specified.
    Parameters
    ----------
        voxel_coords: array
            Needs to contain the boundaries of the voxels. [no. of voxels, 3, 2]
            Coordinate order is xyz, this is the 2nd dimension. Limits are 2, min and max, defined in the 3rd dimesnion.
        survey_coords: array
            Contains the survey location xyz coordinates, [no. of survey locations, 3]
        noise_on_grid: bool
            If True Gaussian noise is added to the survey locations, with std specified in dataset_params.json
        device_noise: bool
            If True Gaussian noise is added to the gravity values, with std specified in dataset_params.json
        background_noise: bool
            If True Gaussian noise is added to the voxel model to simualte noise in the density distribution
        testdata: bool
            If True then the box_params_test.hdf5 is searched for.
            If False then the box_params_train.hdf5 is searched for.
            If netiher are found, then new parameters are generated with datasize given as an input to this method
        filename: str
            The name under which the generated dataset will be saved
        dataloc: str,
            The location where to save the data. (if saving)
        datasize: int,
            The size of the dataset to be made. (ignored if making test data)
        saving: bool
            If True, the dataset is saved, is not, the generated data is only returned as the output
    Output
    ------
        models: array
            The generated set of parameters, which are the voxel densities in this case. [no. of datapoints, no. of voxels]
        surveys: array
            The forward modelled surveys corresponding to each set of parameters. [np. of datapoints, no. of survey locations]
    """
    if saving:
        path_to_datafile = os.path.join(dataloc, filename)
        if os.path.exists(path_to_datafile):
            raise FileExistsError("File with this name already exists.")
    if not os.path.exists('dataset_params.json'):
        raise FileNotFoundError('The dataset_params.json file has to exist.')
    starttime = datetime.now()
    with open('dataset_params.json') as json_file:
        params = json.load(json_file)
    if testdata == True:
        path_to_paramsfile = os.path.join(params['dataloc'], 'box_params_test.hdf5')
    else:
        path_to_paramsfile = os.path.join(params['dataloc'], 'box_params_train.hdf5')
    if os.path.exists(path_to_paramsfile):
        print("Making data from parameters file.")
        h = h5py.File(path_to_paramsfile, "r")
        px = np.array(h['px'])
        py = np.array(h['py'])
        pz = np.array(h['pz'])
        lx = np.array(h['lx'])
        ly = np.array(h['ly'])
        lz = np.array(h['lz'])
        alpha_x = np.array(h['alpha_x'])
        alpha_y = np.array(h['alpha_y'])
        h.close()
        datasize = np.shape(px)[0] # if we are using the new dataset, the data size is inferred from this data set.
        print(f"Data set size inferred from file: {datasize}. Given value overwritten.")
    else:
        print("Data set randomly generated.")
        n_per_side = params['n_per_side']
        widths_grid = params['widths_grid']
        voxel_grid, widths_voxels = make_3D_grid(widths_grid, [n_per_side, n_per_side, n_per_side])
        px, py, pz, lx, ly, lz, alpha_x, alpha_y = make_box_params(voxel_grid, widths_voxels, datasize, saving=False)
        print(f"Data set size given: {datasize}")
    rho = params['delta_rho']
    survey_c = np.zeros(np.shape(survey_coords))
    surveys = []
    models = []
    models_parameterised = []
    for i in range(datasize): # looping over each set of parameters, to simulate the voxelised density distribution and the corresponding survey
        survey_c[:,0] = survey_coords[:,0]
        survey_c[:,1] = survey_coords[:,1]
        survey_c[:,2] = survey_coords[:,2]
        if noise_on_grid:
            if i == 0:
                print(f"Noisy grid. Survey grid noise scale = {params['surveygrid_noise_scale']}")
            survey_c[:,0] = survey_coords[:,0] + np.random.normal(loc=0.0, scale=params['surveygrid_noise_scale'], size=np.shape(survey_c[:,0])[0])
            survey_c[:,1] = survey_coords[:,1] + np.random.normal(loc=0.0, scale=params['surveygrid_noise_scale'], size=np.shape(survey_c[:,1])[0])
            survey_c[:,2] = survey_coords[:,2]
        parameters = [px[i], py[i], pz[i], lx[i], ly[i], lz[i], alpha_x[i], alpha_y[i]]
        # making the voxelised representation of the box
        if background_noise:
            if i == 0:
                print(f"Noisy background density. Density noise scale = {params['density_noise_scale']}")
        model = make_rotated_bunker(voxel_coords, parameters, rho, background_noise=background_noise)
        # the forward model
        gz = get_gz_analytical_vectorised(voxel_coords, model, survey_c)
        if device_noise:
            if i == 0:
                print(f"Noisy survey. Device noise scale = {params['device_noise_scale']}")
            gz = gz + np.random.normal(loc=0.0, scale=params['device_noise_scale'], size=np.shape(gz)[0])
        surveys.append(np.concatenate((np.expand_dims(gz, axis=1), survey_c), axis=1))
        models.append(model)
        models_parameterised.append(np.array(parameters))
        if i % 1000 == 0:
            nowtime = datetime.now()
            print(f"Data generated {i}/{datasize} \t Time elapsed: {nowtime-starttime}")
    surveys = np.array(surveys)
    models = np.array(models)
    models_parameterised = np.array(models_parameterised)
    if saving:
        h = h5py.File(path_to_datafile, "a")
        h.create_dataset("models_parameterised", data=models_parameterised)
        h.create_dataset("models", data=models)
        h.create_dataset("surveys", data=surveys)
        h.create_dataset("noise_on_grid", data=noise_on_grid)
        if noise_on_grid == True:
            h.create_dataset("surveygrid_noise_scale", data=params['surveygrid_noise_scale'])
        h.create_dataset("device_noise", data=device_noise)
        if device_noise == True:
            h.create_dataset("device_noise_scale", data=params['device_noise_scale'])
        h.create_dataset("density_noise_scale", data=params['density_noise_scale'])
        h.create_dataset("voxel_grid", data=voxel_coords)
        h.close()
    return models, surveys

# ------------------------- Box parameter generation, and parameter translation to voxelised density model -------------------------
def make_box_params(voxel_grid, widths_voxels, datasize, dataloc="", filename="box_params.hdf5", saving=True):
    """Makes a dataset of boxes within a specified voxel space. The priors of the parameters are defined here
    Parameters
    ----------
        datasize: int
            The number of parameter sets we want to make.
        voxel_grid: array
           Specifies the locations of the centre each voxel with xyz coordinates. [no. of voxels, 3]
        widths_voxels: list
           The width of voxels in each dimension, each a float. [3]
        saving: bool
           If True, the dataset will be saved in a hdf5 file.
        dataloc: str
           The location where the datafile will be saved.
        filename: str
           The filename where the data is saved.
    Output
    ------
        px: array
           The x coordinate of the centre of the box (the origin is the middle of the defined voxel space) [datasize]
        py: array
           The y coordinate of the centre of the box [datasize]
        pz: array
           The z coordinate of the centre of the box [datasize]
        lx: array
           The length of the box in the x dimension [datasize]
        ly: array
           The length of the box in the y dimension [datasize]
        lz: array
           The length of the box in the z dimension (these are ill-defined, as the box is then rotated) [datasize]
        alpha_x: array
           Parameter defining the rotation, arctan(alpha_x/alpha_y) is the rotation of the box clockwise(?) around its cente around the z axis [datasize]
        alpha_y: array
           Second parameter defining the rotation [datasize]
    """
    if saving:
        path_to_datafile = os.path.join(dataloc, filename)
        if os.path.exists(path_to_datafile):
            raise FileExistsError("File with this name already exists.")
    # specifying the range of possible locations for the box
    x_min = np.min(voxel_grid[:,0]) - widths_voxels[0]/2
    x_max = np.max(voxel_grid[:,0]) + widths_voxels[0]/2
    y_min = np.min(voxel_grid[:,1]) - widths_voxels[1]/2
    y_max = np.max(voxel_grid[:,1]) + widths_voxels[1]/2
    z_min = np.min(voxel_grid[:,2]) - widths_voxels[2]/2
    z_max = (np.max(voxel_grid[:,2]) + widths_voxels[2]/2) - 40 # making sure that the box cannot poke out on the top
    # need to get rid of this hardcoding here
    # location of the middle of the box
    px = np.random.uniform(low=x_min, high=x_max, size=datasize)
    py = np.random.uniform(low=y_min, high=y_max, size=datasize)
    pz = np.random.uniform(low=z_min, high=z_max, size=datasize)
    # its dimensions
    lz_max = z_max - z_min - 40 # making sure that the box cannot poke out on the top
    lz = np.random.uniform(low=0.0, high=lz_max, size=datasize)
    lx_max = x_max-x_min
    lx = np.random.uniform(low=0.0, high=lx_max, size=datasize)
    ly_max = y_max-y_min
    ly = np.random.uniform(low=0.0, high=ly_max, size=datasize)
    # the rotation angle
    # maximum rotation is 180, so only the top two quadrants are taken
    alpha_x = np.random.normal(0, 0.5, size=datasize)
    alpha_y = np.random.normal(0, 0.5, size=datasize)
    # the angle of rotation can be reconstructed such as: alpha = np.arctan(alpha_y/alpha_x)
    if saving:
        f = h5py.File(path_to_datafile, "a")
        f.create_dataset("px", data=px)
        f.create_dataset("py", data=py)
        f.create_dataset("pz", data=pz)
        f.create_dataset("lx", data=lx)
        f.create_dataset("ly", data=ly)
        f.create_dataset("lz", data=lz)
        f.create_dataset("alpha_x", data=alpha_x)
        f.create_dataset("alpha_y", data=alpha_y)
        f.close()
    return px, py, pz, lx, ly, lz, alpha_x, alpha_y

def make_rotated_bunker(voxel_grid, parameters, rho=-1600, background_noise=True):
    """Makes a rotated box with the given parameters [px, py, pz, lx, ly, lz, alpha_x, alpha_y] translated
    to density values at the voxel grid location specified.
    The background density is assumed to be 0.
    Parameters
    ----------
        voxel_grid: array
            Needs to contain the boundaries of the voxels. [number of voxels, 3, 2]
            Coordinate order is xyz given in the 2nd dimension. Limits are 2, min and max, given in the 3rd dimension
        parameters: list
            Contains a set of parameters that define a single box
        rho: float
            The density within the box.
        background_noise: bool
            If True Gaussian background noise is added to density model, with std defined in the params_makedata.json file
    Output
    ------
        model: array
            The voxelised representation of the box defined on the input grid and parameters. [number of voxels]
    """
    if not os.path.exists('dataset_params.json'):
        raise FileNotFoundError('The dataset_params.json file has to exist.')
    with open('dataset_params.json') as json_file:
        params = json.load(json_file)
    num_voxels = np.shape(voxel_grid)[0]
    rho_obj = rho # the relative density within the object
    rho_bg = 0.0
    bg = np.zeros(num_voxels)
    if background_noise:
        density_noise_scale = params['density_noise_scale']
        bg = bg + np.random.normal(loc=0.0, scale=density_noise_scale, size=np.shape(bg)[0])
    px, py, pz, lx, ly, lz, alpha_x, alpha_y = parameters
    alpha = np.arctan(alpha_y/alpha_x)
    # calculating the locations of the vertices of the rectangle, which is the box when viwed from above
    p1 = tuple(rotate(np.array([px-lx/2, py+ly/2]), origin=(px,py), angle=alpha))
    p2 = tuple(rotate(np.array([px+lx/2, py+ly/2]), origin=(px,py), angle=alpha))
    p3 = tuple(rotate(np.array([px-lx/2, py-ly/2]), origin=(px,py), angle=alpha))
    p4 = tuple(rotate(np.array([px+lx/2, py-ly/2]), origin=(px,py), angle=alpha))
    # making the 1d rectangle
    rect1 = Polygon([p1, p2, p4, p3])
    # defining the extent of the box in the z direction
    z_top = pz+lz/2
    z_bottom = pz-lz/2
    densities = []
    for i in range(np.shape(voxel_grid)[0]): # looping over each ovxel
        q1 = [voxel_grid[i,0,0], voxel_grid[i,1,0]] # (x1, y1)
        q2 = [voxel_grid[i,0,0], voxel_grid[i,1,1]] # (x1, y2)
        q3 = [voxel_grid[i,0,1], voxel_grid[i,1,0]] # (x2, y1)
        q4 = [voxel_grid[i,0,1], voxel_grid[i,1,1]] # (x2, y2)
        z1 = voxel_grid[i,2,1] # the limits of the voxel in z, z1 is the max
        z2 = voxel_grid[i,2,0]
        width = z1-z2
        area_voxel = (voxel_grid[i,0,1]-voxel_grid[i,0,0])*(voxel_grid[i,1,1]-voxel_grid[i,1,0])
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
            densities.append(0.0) # if no overlap, the added density at the specific voxel loction is just 0
    densities = np.array(densities)
    parameters = np.array(parameters)
    model = bg+densities
    return model

# --------------------------------------- Making grids ----------------------------------------------
def make_2D_grid(wx, wy, nx, ny, z_surf):
    """Creates a survey grid and returns the coords of the centre of each grid pixel
    Parameters
    ----------
        wx: float
            The width of the survey area in the x dimension
        wy: float
            The width of the survey area in the y dimension
        nx: int
            The number of pixels in the x direction
        ny: int
            The number of pixels in the y direction
    Output
    ------
        coords: array
            Returns the xyz coordinates of the pixel centres. [no. of grid points, 3]
    """
    dx = wx/nx
    dy = wy/ny

    x_cm = np.arange(-wx/2+dx/2, wx/2+dx/2, dx)
    y_cm = np.arange(-wy/2+dy/2, wy/2+dy/2, dy)
    z_cm = z_surf
    X_cm, Y_cm, Z_cm = np.meshgrid(x_cm, y_cm, z_surf, indexing='ij')
    X_cm, Y_cm, Z_cm = X_cm.ravel(), Y_cm.ravel(), Z_cm.ravel()
    coords = np.c_[X_cm, Y_cm, Z_cm]
    return coords

def make_survey_point(range_x, range_y, range_z):
    """Makes a single survey point, picked randomly from the given ranges.
    Parameters
    ----------
        range_x: list
            The range from which the x coordinate is picked [low, high]
        range_y: list
            The range from which the y coordinate is picked.
        range_z: list
            The range from which the z coordinate is picked. If it's a single value, then z is set to that.
    Output
    ------
        survey_coords: array
           The xyz coordinates of the survey point, length 3.
    """
    x = np.random.uniform(range_x[0], range_x[1])
    y = np.random.uniform(range_y[0], range_y[1])
    if len(range_z)==1: # giving the option for z to be a single surface
        z = range_z[0]
    elif len(range_z)==2:
        z = np.random.uniform(range_z[0], range_z[1])
    survey_coords = np.array([x, y, z])
    survey_coords = np.expand_dims(survey_coords, axis=0)
    return survey_coords

def make_3D_grid(widths, nums, centres=np.array([0, 0, 0]), get_volumes=False):
    """Splits up a rectangular volume to a grid of voxels and returns the centres of the voxels.
    Parameters
    ----------
        widths: list
           [wx, wy, wz]: The width of the volume in x, y and z dimension
        nums: list
           [nx, ny, nz]: The number of voxels along each direction
        centres: list
           [cx, cy, cz]: The coords of the centre of the volume
        get_volumes: bool
           If True, the volumes of each voxel are included in the coordinates
    Output
    ------
        coords: array
            The coordinates of each voxel. [number of voxels, 3 (or 4 if volume is returned)]
        widths: array
            Contains the widths of the voxels in the 3 different dimensions, length 3
    """
    wx, wy, wz = widths[0], widths[1], widths[2]
    nx, ny, nz = nums[0], nums[1], nums[2]
    cx, cy, cz = centres[0], centres[1], centres[2]
    dx = wx/nx
    dy = wy/ny
    dz = wz/nz

    x_cm = np.arange(cx-wx/2+dx/2, cx+wx/2+dx/2, dx)
    y_cm = np.arange(cy-wy/2+dy/2, cy+wy/2+dy/2, dy)
    z_cm = np.arange(cz-wz/2+dz/2, cz+wz/2+dz/2, dz)
    X_cm, Y_cm, Z_cm = np.meshgrid(x_cm, y_cm, z_cm, indexing='ij')
    X_cm, Y_cm, Z_cm = X_cm.ravel(), Y_cm.ravel(), Z_cm.ravel()
    widths = np.array([dx, dy, dz])
    if get_volumes:
        V = np.ones(np.shape(X_cm))*dx*dy*dz
        coords = np.c_[X_cm, Y_cm, Z_cm, V] # the volumes are included as the last column
    else:
        coords = np.c_[X_cm, Y_cm, Z_cm]
    return coords, widths

def make_3D_grid_limits(widths, nums, centres=np.array([0.0, 0.0, 0.0])):
    """Creates a grid of voxel locations for a rectangular volume
    Parameters
    ----------
        widths: list
            [wx, wy, wz]: The width of the volume in x, y and z dimension
        nums: list
            [nx, ny, nz]: The number of voxels along each direction
        centres: list
            [cx, cy, cz]: The coords of the centre of the volume
    Output
    ------
        coords: array
            The coordinates of the voxel limits [number of voxels, 3, 2]
            The last dimension is the high and low limits, the 2nd dimension is x y z
    """
    wx, wy, wz = widths[0], widths[1], widths[2]
    nx, ny, nz = nums[0], nums[1], nums[2]
    cx, cy, cz = centres[0], centres[1], centres[2]
    dx = wx/nx
    dy = wy/ny
    dz = wz/nz

    x_min = np.arange(cx-wx/2, cx+wx/2, dx)
    y_min = np.arange(cy-wy/2, cy+wy/2, dy)
    z_min = np.arange(cz-wz/2, cz+wz/2, dz)
    X_min, Y_min, Z_min= np.meshgrid(x_min, y_min, z_min, indexing='ij')
    X_min, Y_min, Z_min = X_min.ravel(), Y_min.ravel(), Z_min.ravel()
    coords_min = np.c_[X_min, Y_min, Z_min]
    coords_min = np.expand_dims(coords_min, axis=2)

    x_max = np.arange(cx-wx/2+dx, cx+wx/2+dx, dx)
    y_max = np.arange(cy-wy/2+dy, cy+wy/2+dy, dy)
    z_max = np.arange(cz-wz/2+dz, cz+wz/2+dz, dz)
    X_max, Y_max, Z_max= np.meshgrid(x_max, y_max, z_max, indexing='ij')
    X_max, Y_max, Z_max = X_max.ravel(), Y_max.ravel(), Z_max.ravel()
    coords_max = np.c_[X_max, Y_max, Z_max]
    coords_max = np.expand_dims(coords_max, axis=2)
    coords = np.concatenate((coords_min, coords_max), axis=2)
    return coords

