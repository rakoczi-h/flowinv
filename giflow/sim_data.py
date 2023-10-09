from datetime import datetime
import json
import numpy as np

from lib.make_data import make_2D_grid, make_3D_grid_limits, make_parameterised_box_dataset, make_voxelised_box_dataset, make_3D_grid, make_box_params
start_time  = datetime.now()

# The parameters of the survey and density voxel grid are given in the params.json file
with open('dataset_params.json') as json_file:
    params = json.load(json_file)

n_per_side = params['n_per_side'] # number of voxels per side = number of survey points per side (they do not have to be the same, this is an arbitrary choice)
widths_survey = np.array(params['widths_survey'])
# the survey is made to span a smaller area than the voxel grid to avoid edge effects
widths_grid = np.array(params['widths_grid'])
z_surf = widths_grid[2]/2 # for now the surface is at the reference level
survey_grid = make_2D_grid(widths_survey[0], widths_survey[1], n_per_side, n_per_side, z_surf)
voxel_grid_limits = make_3D_grid_limits(widths_grid, np.array([n_per_side, n_per_side, n_per_side]))
voxel_grid, widths_voxels = make_3D_grid(widths_grid, np.array([n_per_side, n_per_side, n_per_side]))

dataloc = params['dataloc']
datasize = params['datasize']
testsize = params['testsize']
dataname= params['dataname']
testname= params['testname']

# Generating the file containing the randomised box parameters
make_box_params(voxel_grid, widths_voxels, datasize, dataloc=dataloc, filename="box_params_train")
make_box_params(voxel_grid, widths_voxels, testsize, dataloc=dataloc, filename="box_params_test")

# It is set that there is no inherently added noise on the survey values (this can be added later)
# The survey is set to be taken on a uniform grid, no noise added
test_models, test_surveys = make_parameterised_box_dataset(survey_grid, noise_on_grid=False, device_noise=False, testdata=True, filename=testname+'_parameterised')
models, surveys = make_parameterised_box_dataset(survey_grid, noise_on_grid=False, device_noise=False, testdata=False, filename=dataname+'_parameterised')

end_time = datetime.now()
print(f"Time taken to generate data: {end_time-start_time}")

