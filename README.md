# G.I.Flow
_G.I.Flow_ contains a machine learning tool applying normalising flows to Bayesian gravity inversion. This project has two aspects to it:
1. Generate a data set consisting of rectangular prism (box) voids. Two different representations can be chosen.
   - The box can be described by 7 parameters and the forward model described in [Li et al. (1998)](https://link.springer.com/article/10.1023/A:1006554408567) is applied to compute the gravitational signal at the given gravimetry survey locations.
   - The box can also be described as an array of density values corresponding to a grid of voxels, and the forward model can be computed for each voxel seperately, and the results are summed for all survey locations.
2. Using the generated data set a neural network can be trained which uses a normalising flow method to infer the parameters of the box. Tools to test the network and plot the results are also available.

## Installation

```bash
git clone https://github.com/rakoczi-h/flowinv.git
cd flowinv
```

```bash
conda env create -f environment.yml
conda activate flowenv
```

## Usage
Jupyter notebooks are included that showcase the software tools available.


**Data generation:**
To generate example data sets, follow the steps in `generate_data.ipynb`. This is a walktrough of defining prior distributions for the source parameters and using the built-in classes to create data sets of rectangular prism voids.

**Gravity inversion:**
To train a Normalising Flow on the generated data, follow the steps in `train.ipynb`. This is a walktrough of reading in the training and validation data, defining normalisation methods, setting the hyperparameters and the training procedures, and training the neural network.

**Testing on simulations:**
To test the Normalising Flow on simulated data, follow the steps in `test.ipynb`. This is a walkthrough of how to do a P-P test, generate results from surveys and create posterior probability distribution and source model plots.

## Authors
Henrietta Rakoczi (corresponding author)
Dr Abhinav Prasad
Dr Karl Toland
Dr Christopher Messenger
Prof Giles Hammond

## Acknowledgements
The author acknowledges the use of the [_nflows_](https://github.com/uofgravity/nflows#citing-nflows) (C. Durkan et al. (2019)) package to construct implement the normalising flow elements and the [_glasflow_](https://github.com/uofgravity/glasflow) (Williams et al. (2023)) package to construct the neural network. Oh and definitely also Jollyfant. He is not a real bear by the way.
