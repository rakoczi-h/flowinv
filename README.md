# G.I.Flow
_G.I.Flow_ contains a machine learning tool applying normalising flows to Bayesian gravity inversion. This project has two aspects to it:
1. Generate a data set consisting of rectangular prism (box) underdensities. Two different representations can be chosen.
   - The box can be described by 9 parameters and the forward model described in [Li et al. (1998)](https://link.springer.com/article/10.1023/A:1006554408567) is applied to compute the gravitational signal at the given gravimetry survey locations.
   - The box can also be described as an array of density values corresponding to a grid of voxels, and the forward model can be computed for each voxel seperately, and the results are summed for all survey locations.
2. Using the generated data set a neural network can be trained which uses a normalising flow method to infer the parameters (9 parameteres, or array of densities) of the box. Tools to test the network and plot the results are also included.
The [_nflows_](https://github.com/uofgravity/nflows#citing-nflows) (C. Durkan et al. (2019)) package is used to construct implement the normalising flow elements and [_glasflow_](https://github.com/uofgravity/glasflow) (Williams et al. (2023)) is used to construct the neural network. 

# Requirements
This software only runs on `x84_64/amd64` architectures and requires Python `3.10` or less. Dependencies occupy approximately `4GB` of space.

## Installation
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install nvidia-pyindex
pip install -r requirements.txt
```

## Usage
**Data generation:**
1. make_data.py script includes all functions necessary for the generation of the data set.
2. To generate an example data set, with parameterised data representation, run the script below with an argument defining the location where the data is to be saved.
```bash
python sim_data.py /path/to/data/
```
The inputs to the data generation can be edited in the *dataset_params.json* file.

**Gravity inversion:**
1. A neural network can be trained with the generated data set, or any other data set we desire to use. The network parameters are defined in *flow_params.json* and the input data and output directory need to be passed as inputs to the script.

```bash
python grav_inv_flow.py /path/to/data/file.hdf5 /output/directory/
```
3. During training, diagnostics can be plotted and saved.
4. After training, the trained flow model can be saved, which then can be used to test the performance of the method or simply use it for inversion. As long as the problem we are trying to solve is consistent with the training data set, the network does not need to be retrained.
To run a test with the example set of data and a pre-trained network model:
```bash
```
The test data location and flow model locations are defined in _params.json_.
## Contributing

## Acknowledgements
