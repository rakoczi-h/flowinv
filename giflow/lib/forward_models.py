import itertools
import numpy as np

# ------------------------------------- Analytical forward model ---------------------------------------------
def get_gz_analytical(limits, rho, survey_loc):
    """
    Jung (1961) and Plouff (1966), review Li (1998)
    Calculates the gravitational anomaly at the survey locations from a single rectangular prism.
    Its faces have to be aligned with the xyz axis. If we want to rotate the box, the survey locations need to be rotated before given to this function.
    param: limits: array [coordinates, limits]
           rho: density within the box
           survey_loc: array, [number of points, coordinates]
    output: gz: [number of points] array of gravity values in microGal.
    """
    G = 6.6743*1e-11
    xs = [survey_loc[:,0]-limits[0,0], survey_loc[:,0]-limits[0,1]]
    ys = [survey_loc[:,1]-limits[1,0], survey_loc[:,1]-limits[1,1]]
    zs = [survey_loc[:,2]-limits[2,0], survey_loc[:,2]-limits[2,1]]
    c = [1,2]
    coefs = np.array(list(itertools.product(c,c,c)))
    mu = np.power(-1, coefs[:,0])*np.power(-1, coefs[:,1])*np.power(-1, coefs[:,2])
    mu = np.repeat(np.expand_dims(mu, axis=1), np.shape(survey_loc)[0], axis=1)
    n = np.shape(coefs)[0]
    terms = np.array(list(itertools.product(xs, ys, zs))) # the last axis is the different survey locations
    r = np.sqrt(terms[:,0,:]**2+terms[:,1,:]**2+terms[:,2,:]**2)
    eq = mu*(terms[:,0,:]*np.log(terms[:,1,:]+r) + terms[:,1,:]*np.log(terms[:,0,:]+r) - terms[:,2,:]*np.arctan(terms[:,0,:]*terms[:,1,:]/(terms[:,2,:]*r)))
    gz = -G*rho*np.sum(eq, axis=0)*1e8 # in microGal
    return gz

def get_gz_analytical_vectorised(limits, rho, survey_loc):
    """
    Jung (1961) and Plouff (1966), review Li (1998)
    A vectorised version of get_gz_analytical. Calculates the gravitational signature due to multiple boxes,
    and sums the values in the end. Useful for voxelised models.
    param: limits: array, [number of voxels, coordinates, limits] coordinate order is xyz, and limits are 2, min and max
           survey_loc: array, [number of points, coordinates], coordinate order is xyz
           rho: vector of floats with size [number_voxels]
    output: gz: [number of points] array of gravity values in microGal, at the given survey locations
    """
    if np.shape(rho)[0] != np.shape(limits)[0]:
        print("Sizes of density and voxel vectors do not match.")
        exit()
    G = 6.6743*1e-11
    survey_loc = np.repeat(np.expand_dims(survey_loc, axis=0), np.shape(limits)[0], axis=0) #[num_voxels, num_surveys, 3]
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
    # fixing zeros in a dodgy way
    logterm1 = terms[:,1,:,:]+r
    logterm1[np.where(logterm1==0)]=1e-100
    logterm2 = terms[:,0,:,:]+r
    logterm2[np.where(logterm2==0)]=1e-100
    divterm = terms[:,2,:,:]*r
    divterm[np.where(divterm==0)]=1e-100
    eq = mu*(terms[:,0,:,:]*np.log(logterm1) + terms[:,1,:,:]*np.log(logterm2) - terms[:,2,:,:]*np.arctan((terms[:,0,:,:]*terms[:,1,:,:])/(divterm)))
    rho = np.repeat(np.expand_dims(rho, axis=1), np.shape(xs)[2], axis=1)
    gz = -G*rho*np.sum(eq, axis=0)*1e8 # in microGal, summing the terms in the equation for a single voxel
    gz = np.sum(gz, axis=0) # summing the contribution from each voxel
    return gz

# ------------------------------------------ Method applying numerical integration --------------------------------------------
# Not recommended to use
def get_gz_numint(object_coords, survey_coords, rho=1):
    """
    THIS USES A SINGLE DENSITY VALUE FOR ALL VOXELS!
    Calculate the z component of the overall gravitational field from all voxels at each survey location.
    param: object_coords: The cartesian coordinates of the com of each voxel in the cylinder/box, and their volume
           survey_coords: The cartesian coordinates of each survey location
           rho: the density of the object, this is a single density for all the voxels considered
    output: the coordinates of the survey points and the corresponding g_z values in uGal
    """
    # Taking the bg to be 0 and the density difference to be the absolute density
    G = 6.6743*1e-11 # m^3kg^-1s^-2
    gravz = []
    for i in range(np.shape(survey_coords)[0]):
        gz = 0
        for j in range(np.shape(object_coords)[0]):
            x = np.abs(survey_coords[i,0]-object_coords[j,0])
            y = np.abs(survey_coords[i,1]-object_coords[j,1])
            z = np.abs(survey_coords[i,2]-object_coords[j,2])
            v = object_coords[j,3]
            r = np.sqrt(x**2+y**2+z**2)
            cos_th = z/r
            gz_j = (-G*v*rho/r**2)*cos_th
            gz = gz+gz_j
        gravz.append(gz*1e8) # changing to microGal
    gravz = np.expand_dims(np.array(gravz), axis=1)
    return np.concatenate((survey_coords, gravz), axis=1)

def forw_numint(model, voxel_coords, survey_coords, widths):
    gravz = np.zeros(np.shape(survey_coords)[0])
    for i in range(np.shape(voxel_coords)[0]):
        coords, _ = make_3D_grid(widths=widths, nums=[5,5,5], centres=voxel_coords[i,:], get_volumes=True) # the nums coordinate controls how sophisticated the model is
        gz = get_gz_numint(coords, survey_coords, rho=model[i]) #getting the gravitational field from a single cube
        gravz = gravz + gz[:, 3]
    return gravz

