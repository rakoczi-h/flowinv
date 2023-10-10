import itertools
import numpy as np

# ------------------------------------- Analytical forward model ---------------------------------------------
def get_gz_analytical(limits, rho, survey_loc):
    """
    Jung (1961) and Plouff (1966), review Li (1998)
    Calculates the gravitational anomaly at the survey locations from a single rectangular prism.
    Its faces have to be aligned with the xyz axis. If we want to rotate the box, the survey locations need to be rotated before given to this function.
    Parameters
    ----------
        limits: array
            The limits of the box in 3 different coordinate dimensions. [3, 2]
        rho: float
            Density within the box/ density contrast with the background.
        survey_loc: array
            The array containing the survey points. [number of points, coordinates]
    Output
    ------
        gz: array
            [number of points] Array of gravity values in microGal.
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
    Parameters
    ----------
        limits: array
            The limits of the voxels in the 3 different coordinate dimensions [number of voxels, 3, 2]
        rho: array
            Array containing the density values for each voxel. [number_voxels]
        survey_loc: array
            The survey locations [number of points, coordinates]
    Parameters
    ----------
        gz: array
            [number of points] Array of gravity values in microGal, at the given survey locations
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


