# G.I.Flow
_G.I.Flow_ contains a machine learning tool applying normalising flows to Bayesian gravity inversion. This project has two aspects to it:
1. Generate a data set consisting of rectangular prism (box) underdensities. Two different representations can be chosen.
   - The box can be described by 9 parameters and the forward model described in [Li et al. (1998)](https://link.springer.com/article/10.1023/A:1006554408567) is applied to compute the gravitational signal at the given gravimetry survey locations.
   - The box can also be described as an array of density values corresponding to a grid of voxels, and the forward model can be computed for each voxel seperately, and the results are summed for all survey locations.
2. Using the generated data set a neural network can be trained which uses a normalising flow method to infer the parameters (7 parameteres, or array of densities) of the box. Tools to test the network and plot the results are also included.

## Installation

```bash
git clone https://github.com/rakoczi-h/giflow.git
cd giflow
```

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install nvidia-pyindex
pip install -r requirements.txt
```

## Usage
**Data generation:**
To generate an example data set, with parameterised data representation, run
```bash
python make_box_dataset.py
```
The data will be saved in */data* in the root folder.

**Gravity inversion:**
A neural network can be trained with the generated data set from above, or any other data set we desire to use. The example script can be ran as:

```bash
python train.py
```
The output is saved in */results* in the root folder.

During training, diagnostics can be plotted and saved.
![Alt text](/fig/diagnostics.png "Diagnostics")

After training, the trained flow model is saved, which then can be used to test the performance of the method or simply use it for inversion. As long as the problem we are trying to solve is consistent with the training data set, the network does not need to be retrained.
To run a test with the example set of data and a pre-trained network model:

```bash
python test.py
```

## Contributing

## Acknowledgements
The author acknowledges the use of the [_nflows_](https://github.com/uofgravity/nflows#citing-nflows) (C. Durkan et al. (2019)) package to construct implement the normalising flow elements and the [_glasflow_](https://github.com/uofgravity/glasflow) (Williams et al. (2023)) package to construct the neural network. 
