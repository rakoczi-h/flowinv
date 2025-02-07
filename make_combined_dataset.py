import pickle as pkl
import os
from giflow.box import Box

data_loc = '/scratch/balta1/2263373r/box/narrow_volume/voxelised_noisy/'
save_loc = '/scratch/balta1/2263373r/box/narrow_volume/combined/'

#filenames = ['trainset_2.pkl', 'trainset_3.pkl', 'trainset_4.pkl','trainset_5.pkl','trainset_6.pkl','trainset_7.pkl','trainset_8.pkl','trainset_9.pkl', 'validationset_2.pkl', 'validationset_3.pkl', 'validationset_4.pkl']
filenames = [f"validationset_{n}.pkl" for n in range(2, 5)]
for filename in filenames:
    with open(os.path.join(data_loc, filename), 'rb') as file:
        dt = pkl.load(file)
    for b in dt.boxes:
        b.translate_to_parameterised_model()
        b.voxelised_model = None
        b.voxel_grid = None
    dt.model_framework['type'] = 'parameterised'
    with open(os.path.join(save_loc, filename), 'wb') as file:
        pkl.dump(dt, file)
