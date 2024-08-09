import numpy as np
from scipy.spatial.distance import jensenshannon
import scipy.stats
from sklearn.preprocessing import MinMaxScaler
import corner
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.lines as mlines
import os
import matplotlib.gridspec as gridspec
import torch
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from .box import Box
from .prior import Prior
from .survey import GravitySurvey
from .plot import make_gif

import plotly.io as pio
pio.templates.default = "plotly_white"


class FlowResults:
    """
    Class to incorporate methods of testing and visualising the results from a NF inversion.
    All parameters are assumed to have been rescaled to their original values. 
    Parameters
    ----------
        samples: np.ndarray
            An array of the samples from the flow, these contain parameter values. [num of samples, num of parameters]
        conditional: np.ndarray or torch.Tensor
            The conditional based on which these samples were generated. Essentially the gravity survey we're inverting
            If it is a torch tensor, it is converted to an np.ndarray
        survey_coordinates: np.ndarray
            The coordnates associated with the gravity data. [num of survey points, 3] (x, y, z is the order)
        log_probabilities: np.ndarray
            The log probability associated with each sample
        parameter_labels: list
            list of string containing the names of the inferred parameters
        true_parameters: np.ndarray
            The true values of the parameters, if known
        directory: str
            The location where the plots of the results are generated
    """
    def __init__(self, samples : np.ndarray, conditional, survey_coordinates=None, log_probabilities=None, parameter_labels=None, true_parameters=None, directory=None):
        self.nparameters = np.shape(samples)[1]
        self.nsamples = np.shape(samples)[0]
        if samples.ndim != 2:
            raise ValueError ('samples has to be 2D.')
        self.samples = samples
        self.conditional = conditional
        self.survey_coordinates = survey_coordinates
        self.parameter_labels = parameter_labels
        self.true_parameters = true_parameters
        self.log_probabilities = log_probabilities
        self.directory = directory

    def __setattr__(self, name, value):
        if name == 'log_probabilities':
            if value is not None:
                if self.nsamples != np.shape(value)[0]:
                    raise ValueError('The same number of samples and log_probabilities are required.')
                elif value.ndim != 1:
                    raise ValueError('log_probabilities has to be 1D.')
        if name == 'parameter_labels':
            if value is not None:
                if len(value) != self.nparameters:
                    print('Same number of labels are required are nparameters. Ignoring labels.')
                    value = None
        if name == 'directory':
            if value is not None:
                if not isinstance(value, str):
                    raise ValueError("Expected str for directory name")
                if not os.path.exists(value):
                    os.mkdir(value)
        if name == 'conditional':
            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
        super().__setattr__(name, value)


    def get_js_divergence(self, samples_to_compare, n=500):
        """Function calculating the Jensen-Shannon divergence between the distribution of the samples of this class and another set of samples.
        The p(x) and q(x) functions are calculated using a KDE of the input samples.
        This is done for each dimension seperately.
        Parameters
        ----------
            samples_to_compare: array
                Samples from the other sampler. [no. of samples, no. of dimensions]. Assumed to be in the original data space (not normalised)
            n: int
                The number of gridpoints to consider when computing the kdes
        Output
        ------
            js: array of floats
                The list of JS-divergence values with length of the no. of parameters/dimensions.
        """
        if self.samples is None:
            if scaler is not None:
                self.inverse_scale('samples', scaler)
            else:
                raise AttributeError("Samples have not been unnormalised and scaler was not provided.")
        if not self.nparameters == np.shape(samples_to_compare)[1]:
            raise ValueError('The two sample sets do not have the same number of parameters.')
        js = []
        for i, dim in enumerate(self.samples.T):
            xmin = min([np.min(dim), np.min(samples_to_compare[:self.nsamples,i])])
            xmax = max([np.max(dim), np.max(samples_to_compare[:self.nsamples,i])])
            # calculate the minimum and maximum from both
            x_grid = np.arange(xmin, xmax+((xmax-xmin)/n), (xmax-xmin)/n) # the grid values we are using
            p = scipy.stats.gaussian_kde(dim)
            p_x = p.evaluate(x_grid)
            q = scipy.stats.gaussian_kde(samples_to_compare[:,i])
            q_x = q.evaluate(x_grid)
            js_pq = np.nan_to_num(np.power(jensenshannon(p_x, q_x), 2))
            js.append(js_pq)
        js = np.array(js)
        return js


    def corner_plot(self, filename='corner.png', prior_bounds=None):
        """Makes a simple corner plot with a single set of posterior samples.
        Parameter
        ---------
        filename: str
            The name under which it is saved
        scaler: sklearn.preprocessing scaler object
            Only used if self.samples does not exist.
        priors_bounds: list
            The length of the list is the same as the dimensions, and each element in the list is [minimum, maximum] bounds.
        """
        plot_range = []
        if prior_bounds is None:
            for dim in self.samples.T:
                plot_range.append([min(dim), max(dim)])
        else:
            plot_range = prior_bounds
        if self.parameter_labels is None:
            labels = [f"q{x}" for x in range(self.nparameters)]
        else:
            labels = self.parameter_labels
        CORNER_KWARGS = dict(smooth=0.9,
                            show_titles=True,
                            label_kwargs=dict(fontsize=20),
                            title_kwargs=dict(fontsize=20),
                            quantiles=[0.16, 0.5, 0.84],
                            levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
                            plot_density=False,
                            plot_datapoints=False,
                            fill_contours=True,
                            max_n_ticks=3,
                            range=plot_range,
                            labels=labels)

        figure = corner.corner(self.samples, **CORNER_KWARGS, color='#ff7f00')
        if self.true_parameters is not None:
            values = self.true_parameters[0]
            corner.overplot_lines(figure, values, color="black")
            corner.overplot_points(figure, values[None], marker="s", color="black")
        if self.directory is not None:
            plt.savefig(os.path.join(self.directory, filename), transparent=False)
        else:
            plt.savefig(filename, transparent=False)
        plt.close()
        print("Made corner plot...")

    # fix overlaid corners method !!
    def overlaid_corner(self, other_samples, dataset_labels, parameter_labels = None, filename='corner_plot_compare',  prior_bounds=None):
        """
        Plots multiple corners on top of each other
        Parameters
        ----------
            samples_list: list of arrays
                Contains samples from different inference algorithms
            parameter_labels: list
                The labels of the parameters over which the posterior is defined
            dataset_labels: list
                The name of the different methods the samples come from
            values: list
                The values of the true parameters, if not None then it is plotted over the posterior
            saveloc: str
                Location where the image is saved
            filename: str
                The name under which it is saved
            priors_bounds: list
                The length of the list is the same as the dimensions, and each element in the list is [minimum, maximum] bounds.
        Output
        ------
        image file
        """
        _, ndim = other_samples.shape
        colors = ['#377eb8', '#ff7f00']

        n = 2
        samples_list = [other_samples, self.samples]
        max_len = max([len(s) for s in samples_list])
        plot_range = []
        if prior_bounds is None:
            for dim in range(ndim):
                plot_range.append(
                    [
                        min([min(samples_list[i].T[dim]) for i in range(n)]),
                        max([max(samples_list[i].T[dim]) for i in range(n)]),
                    ]
                )
        else:
            plot_range = prior_bounds
        if parameter_labels is None:
            if self.parameter_labels is None:
                labels = [f"q{x}" for x in range(self.nparameters)]
            else:
                labels = self.parameter_labels
        else:
            labels = parameter_labels

        CORNER_KWARGS = dict(
        smooth=0.9,
        show_titles=True,
        label_kwargs=dict(fontsize=20),
        title_kwargs=dict(fontsize=20),
        quantiles=[0.16, 0.5, 0.84],
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
        plot_density=False,
        plot_datapoints=False,
        fill_contours=True,
        max_n_ticks=3,
        range=plot_range,
        labels=labels)

        fig = corner.corner(
            samples_list[0],
            color=colors[0],
            **CORNER_KWARGS,
            hist_kwargs={'density' : True}
        )

        for idx in range(1, n):
            fig = corner.corner(
                samples_list[idx],
                fig=fig,
                weights=np.ones(len(samples_list[idx]))*(max_len/len(samples_list[idx])),
                color=colors[idx],
                **CORNER_KWARGS,
                hist_kwargs={'density' : True}
            )
        if self.true_parameters is not None:
            values = self.true_parameters[0]
            corner.overplot_lines(fig, values, color="black")
            corner.overplot_points(fig, values[None], marker="s", color="black")
        plt.legend(
            handles=[
                mlines.Line2D([], [], color=colors[i], label=dataset_labels[i])
                for i in range(n)
            ],
            fontsize=20, frameon=False,
            bbox_to_anchor=(1, ndim), loc="upper right"
        )
        if self.directory is not None:
            plt.savefig(os.path.join(self.directory, filename), transparent=False)
        else:
            plt.savefig(filename, transparent=False)
        plt.close()
        print("Made corner plot...")

class BoxFlowResults(FlowResults):
    """
    Child class of FlowResults for specifically handling visualisation and processing of results from inversion concerning boxes. 
    """
    def rescale(self, scale_factor, parameters_to_rescale=[]):
        for i, pl in enumerate(self.parameter_labels):
            if pl in parameters_to_rescale:
                self.samples[:,i] = self.samples[:,i]*scale_factor

    def plot_compare_surveys(self, model_framework, survey_framework=None, num=1000, include_examples=False, filename='compare_survey.png'):
        """
        Forward models the samples from the flow and compares the forward mdoel to the input.
        Parameters
        ----------
            model_framework: dict
                The BoxDataSet attribute can just directly be passed to this.
                Has to have keys 'ranges': list of 3 values, and 'grid_shape': list of 3 values, 'density': float
            survey_framework: dict
                The BoxDataSet attribute can be passed to this
                Has to have keys 'noise_scale': float, 'ranges': [[],[],[]], 'survey_shape': float or list
            num: int
                Number of samples to use
            include_examples: bool
                Whether to plot a few individual samples.
        """
        if num > np.shape(self.samples)[0]:
            num = np.shape(self.samples)[0]
            print("Not enough samples, using {num} samples only.")

        if self.survey_coordinates is None:
            if survey_framework is not None:
                survey = GravitySurvey(ranges=survey_framework['ranges'], noise_scale=survey_framework['noise_scale'], survey_shape=survey_framework['survey_shape'])
                survey.make_survey()
                coordinates = survey.survey_coordinates
            else:
                raise ValueError("Either provide the survey_coordinates attribute to the class or give the survey_framework as an input to the function")
        else:
            coordinates = self.survey_coordinates

        #num_survey_points = np.shape(coordinates)[0]
        #num_survey_coordinates = int(np.shape(self.conditional)[0]/num_survey_points)
        target_array = np.array(self.conditional[0])
        target = target_array
        #target = target_array - np.mean(target_array)

        mode = model_framework['type']
        gzs = []
        for i in range(num):
            if mode == 'parameterised':
                box = Box(parameterised_model=self.samples[i,:], parameter_labels=self.parameter_labels, density=model_framework['density'])
                box.translate_to_parameters()
            elif mode == 'voxelised':
                box = Box(voxelised_model=self.samples[i,:])
                box.make_voxel_grid(ranges=model_framework['ranges'], grid_shape=model_framework['grid_shape'])
            #gz = box.forward_model(survey_coordinates=coordinates.copy(), model_type=mode)-np.mean(target_array)
            gz = box.forward_model(survey_coordinates=coordinates.copy(), model_type=mode)
            if any(np.isnan(gz)):
                print("Found NaN in simualted gravity from sample. Removing sample")
                continue
            if any(np.isinf(gz)):
                print("Found inf in simualted gravity from sample. Removing sample")
                continue
            gzs.append(gz)
        gzs = np.array(gzs)
        mean = np.mean(gzs, axis=0)
        std = np.std(gzs, axis=0)

        plot_data = [target, mean, std, gzs[0,:], gzs[1,:], gzs[2,:], gzs[3,:], gzs[4,:], gzs[5,:]]
        titles = ['Target', 'Mean', 'Std', 'Sample 1', 'Sample 2', 'Sample 3', 'Sample 4', 'Sample 5', 'Sample 6']
        if include_examples:
            fig, axes = plt.subplots(nrows=3, ncols=3)
        else:
            fig, axes = plt.subplots(nrows=1, ncols=3)
        vmin = np.array([target.min(), mean.min()]).min()
        vmax = np.array([target.max(), mean.max()]).max()
        levels = np.linspace(vmin, vmax, 15)
        cmap = 'plasma'
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for idx, ax in enumerate(axes.flatten()):
            print(titles[idx])
            ax.plot(coordinates[:,0], coordinates[:,1], 'o', markersize=2, color='black')
            ax.tricontourf(coordinates[:,0], coordinates[:,1], plot_data[idx], levels=levels, cmap=cmap, norm=norm)
            ax.set(xlim=(np.min(coordinates[:,0]), np.max(coordinates[:,0])), ylim=(np.min(coordinates[:,1]), np.max(coordinates[:,1])), aspect='equal', title=titles[idx])
        cax = ax.inset_axes([1.1, 0.0, 0.1, 3.35])
        plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label=r'microGal')
        fig.tight_layout()
        if self.directory is not None:
            plt.savefig(os.path.join(self.directory, filename), transparent=False)
        else:
            plt.savefig(filename, transparent=False)
        plt.close()

    def plot_compare_voxel_slices(self, slice_coords=[1,3,5], filename='sliced_voxels.png', plot_truth=False, normalisation=None, model_framework=None):
        """Makes a comparison plot consisting of slices of the voxelspace.
        Each column is slices along a different direction (x, y, z).
        Each row is a different slice, with increasing coordinates.
        The method is made for 3 slices.
        Parameters
        ----------
            slice_coords: list
                The coordinate of voxels along which to slice the volume. Has to have length 3.
            filename: str
                The name of the file under which it will be saved.
            plot_truth: bool
                Defines whether the true voxelised model is added to the plot.
            normalisation: list
                If not None, the list has to be two elements long, and it defines the color normalisation. [minimum value of color scaler, maximum value]
            model_framework: dict
                If None, then the samples are directly plotted and assumed that each value is a density of a voxel on a grid. If it is given, then the dictionary is assumed to contain values that can be passed to the Box class model_framework attribute.
        """
        print(np.shape(self.samples))
        if normalisation is not None:
            if len(normalisation) != 2:
                raise ValueError('The normalisation input needs to be a list with 2 elements, defining the minimum and maximum of the color scale')
        if len(slice_coords) != 3:
                raise ValueError('Only three slices can be defined')
        if plot_truth:
            if self.true_parameters is None:
                raise ValueError("Give the model as the true_parameters attribute to the class")
            true_model = self.true_parameters[0,:]

        if model_framework is not None:
            if model_framework['type'] == 'parameterised':
                print("Translating samples to voxelised model...")
                if self.parameter_labels is None:
                    raise ValueError('Provide the parameter labels as an attribute to the Results class.')
                samples = []
                for s in self.samples:
                    box = Box(parameterised_model=s, parameter_labels=self.parameter_labels)
                    box.translate_to_parameters()
                    box.make_voxel_grid(model_framework['grid_shape'], model_framework['ranges'])
                    box.translate_to_voxels(density=model_framework['density'])
                    samples.append(box.voxelised_model)
                samples = np.vstack(samples)
                print(np.shape(samples))
                if plot_truth:
                    box = Box(parameterised_model=true_model, parameter_labels=self.parameter_labels)
                    box.translate_to_parameters()
                    box.make_voxel_grid(model_framework['grid_shape'], model_framework['ranges'])
                    box.translate_to_voxels(density=model_framework['density'])
                    true_model = box.voxelised_model
            if model_framework['type'] == 'voxelised':
                samples = self.samples
        else:
            samples = self.samples
        print(np.shape(samples))

        d = round(np.power(np.shape(samples[0,:])[0], 1/3))
        s1, s2, s3 = slice_coords
        if plot_truth:
            shift_idx = 0
            plot_data = np.zeros((9, len(slice_coords)+1, d, d)) # [number of subfigures, number of subplots, dim1, dim2]
            # Plotting the true slices
            true_model = np.flip(np.reshape(true_model, (d,d,d), order='F'))
            #true_model = np.reshape(true_model, (d,d,d)
            plot_data[0, 0, :, :] = np.rot90(true_model[s1, :, :], axes=(0,1), k=3)
            plot_data[3, 0, :, :] = np.rot90(true_model[s2, :, :], axes=(0,1), k=3)
            plot_data[6, 0, :, :] = np.rot90(true_model[s3, :, :], axes=(0,1), k=3)

            plot_data[1, 0, :, :] = np.flip(true_model[:, s3, :], axis=1)
            plot_data[4, 0, :, :] = np.flip(true_model[:, s2, :], axis=1)
            plot_data[7, 0, :, :] = np.flip(true_model[:, s1, :], axis=1)

            plot_data[2, 0, :, :] = np.flip(true_model[:, :, s3], axis=1)
            plot_data[5, 0, :, :] = np.flip(true_model[:, :, s2], axis=1)
            plot_data[8, 0, :, :] = np.flip(true_model[:, :, s1], axis=1)
        else:
            shift_idx = 1
            plot_data = np.zeros((9, len(slice_coords), d, d))

        # Mean
        mean_model = np.mean(samples, axis=0)
        mean_model = np.flip(np.reshape(mean_model, (d,d,d), order='F'))
        plot_data[0, 1-shift_idx, :, :] = np.rot90(mean_model[s1, :, :], axes=(0,1), k=3)
        plot_data[3, 1-shift_idx, :, :] = np.rot90(mean_model[s2, :, :], axes=(0,1), k=3)
        plot_data[6, 1-shift_idx, :, :] = np.rot90(mean_model[s3, :, :], axes=(0,1), k=3)

        plot_data[1, 1-shift_idx, :, :] = np.flip(mean_model[:, s3, :], axis=1)
        plot_data[4, 1-shift_idx, :, :] = np.flip(mean_model[:, s2, :], axis=1)
        plot_data[7, 1-shift_idx, :, :] = np.flip(mean_model[:, s1, :], axis=1)

        plot_data[2, 1-shift_idx, :, :] = np.flip(mean_model[:, :, s3], axis=1)
        plot_data[5, 1-shift_idx, :, :] = np.flip(mean_model[:, :, s2], axis=1)
        plot_data[8, 1-shift_idx, :, :] = np.flip(mean_model[:, :, s1], axis=1)

        # Mode
        mode_model = samples[np.argmax(self.log_probabilities), :]
        mode_model = np.flip(np.reshape(mode_model, (d,d,d), order='F'))
        plot_data[0, 2-shift_idx, :, :] = np.rot90(mode_model[s1, :, :], axes=(0,1), k=3)
        plot_data[3, 2-shift_idx, :, :] = np.rot90(mode_model[s2, :, :], axes=(0,1), k=3)
        plot_data[6, 2-shift_idx, :, :] = np.rot90(mode_model[s3, :, :], axes=(0,1), k=3)

        plot_data[1, 2-shift_idx, :, :] = np.flip(mode_model[:, s3, :], axis=1)
        plot_data[4, 2-shift_idx, :, :] = np.flip(mode_model[:, s2, :], axis=1)
        plot_data[7, 2-shift_idx, :, :] = np.flip(mode_model[:, s1, :], axis=1)

        plot_data[2, 2-shift_idx, :, :] = np.flip(mode_model[:, :, s3], axis=1)
        plot_data[5, 2-shift_idx, :, :] = np.flip(mode_model[:, :, s2], axis=1)
        plot_data[8, 2-shift_idx, :, :] = np.flip(mode_model[:, :, s1], axis=1)

        # Std
        std_model = -np.std(samples, axis=0)
        std_model = np.nan_to_num(std_model)
        std_model = np.flip(np.reshape(std_model, (d,d,d), order='F'))
        plot_data[0, 3-shift_idx, :, :] = np.rot90(std_model[s1, :, :], axes=(0,1), k=3)
        plot_data[3, 3-shift_idx, :, :] = np.rot90(std_model[s2, :, :], axes=(0,1), k=3)
        plot_data[6, 3-shift_idx, :, :] = np.rot90(std_model[s3, :, :], axes=(0,1), k=3)

        plot_data[1, 3-shift_idx, :, :] = np.flip(std_model[:, s3, :], axis=1)
        plot_data[4, 3-shift_idx, :, :] = np.flip(std_model[:, s2, :], axis=1)
        plot_data[7, 3-shift_idx, :, :] = np.flip(std_model[:, s1, :], axis=1)

        plot_data[2, 3-shift_idx, :, :] = np.flip(std_model[:, :, s3], axis=1)
        plot_data[5, 3-shift_idx, :, :] = np.flip(std_model[:, :, s2], axis=1)
        plot_data[8, 3-shift_idx, :, :] = np.flip(std_model[:, :, s1], axis=1)

        if normalisation is None:
            norm = plt.cm.colors.Normalize(np.min(mean_model), np.max(mean_model))
        else:
            norm = plt.cm.colors.Normalize(normalisation[0], normalisation[1])
        cmap = 'plasma'

        fig = plt.figure(figsize=(16, 14))
        outer = gridspec.GridSpec(3, len(slice_coords), wspace=0.2, hspace=-0.79)
        ylabels = ['y', 'z', 'z',
                   'y', 'z', 'z',
                   'y', 'z', 'z']
        xlabels = ['x', 'y', 'x',
                   'x', 'y', 'x',
                   'x', 'y', 'x']
        for i in range(int(3*len(slice_coords))):
            if plot_truth:
                r = len(slice_coords)+1
                inner = gridspec.GridSpecFromSubplotSpec(1, r, subplot_spec=outer[i],
                                                     wspace=0.1, hspace=0.1)
            else:
                r = len(slice_coords)
                inner = gridspec.GridSpecFromSubplotSpec(1, r, subplot_spec=outer[i],
                                                     wspace=0.1, hspace=0.1)
            row     = 0
            col     = 0
            maxCol  = 4

            for j in range(r):
                ax = plt.Subplot(fig, inner[j])
                im = ax.imshow(plot_data[i, j, :, :], norm=norm, cmap=cmap, aspect='equal')
                ax.set_xticks([])
                ax.set_yticks([])
                if i < 3:
                    if plot_truth:
                        if j == 0:
                            ax.set_title('Target', fontsize=14)
                            ax.set_ylabel(ylabels[i], fontsize=14)
                            ax.set_xlabel(xlabels[i], fontsize=14)
                    if j == 1-shift_idx:
                        ax.set_title("Mean", fontsize=14)
                        if not plot_truth:
                            ax.set_ylabel(ylabels[i], fontsize=14)
                            ax.set_xlabel(xlabels[i], fontsize=14)
                    if j == 2-shift_idx:
                        ax.set_title('Mode', fontsize=14)
                    if j == 3-shift_idx:
                        ax.set_title('SD', fontsize=14)
                else:
                    if j == 0:
                        ax.set_ylabel(ylabels[i], fontsize=14)
                        ax.set_xlabel(xlabels[i], fontsize=14)
                fig.add_subplot(ax)

        cbar_ax = fig.add_axes([0.91, 0.35, 0.015, 0.29])
        fig.colorbar(im, cax=cbar_ax, cmap=cmap, norm=norm)
        cbar_ax.set_ylabel(f"\u03C1 [kg/$m^{3}$]",fontsize=14)
        cbar_ax.tick_params(labelsize=14)

        plt.savefig(os.path.join(self.directory, filename), bbox_inches='tight', transparent=True)
        plt.close()

    def plot_3D_statistics(self, model_framework, filename='3D_statistics.html', axis_scale=None):
        """
        Creates a 3D plot. Either can plot statistics, or can plot an animated gif of samples.
        Parameters
        ----------
            model_framework: dict
                Dictionary containing information about the box model setup.
            filename: str
                The location where the image is saved.
            axis_scale: float
                The coordinates of the voxel grid are scaled based on this value.

        """
        box = Box()
        box.make_voxel_grid(model_framework['grid_shape'], model_framework['ranges'])
        x = np.mean(box.voxel_grid[:,0,:], axis=1)
        y = np.mean(box.voxel_grid[:,1,:], axis=1)
        z = np.mean(box.voxel_grid[:,2,:], axis=1)

        if axis_scale is not None:
            x, y, z= x*axis_scale, y*axis_scale, z*axis_scale

        titles = ["Mean", "Mode", "Std", "Fractional Std"]
        mean = np.mean(self.samples, axis=0)
        mean = mean - model_framework['density']
        mode = self.samples[np.argmax(self.log_probabilities), :]
        mode = mode - model_framework['density']
        std = np.std(self.samples, axis=0)
        std_frac = std/mean
        models = [mean, mode, std, std_frac]

        if self.true_parameters is not None:
            titles.append("Truth")
            truth = self.true_parameters.flatten() - model_framework['density']
            models.append(truth)

        for i, m in enumerate(models):
            title = titles[i]
            fig = go.Figure(data=go.Volume(x=x, y=y, z=z, value=m, colorscale='Plasma', opacity=0.7, opacityscale='min', surface_count=17))
            fig.update_layout(height = 800,
                            width = 1000,
                            font = dict(size=12),
                            scene = dict(
                               xaxis_title='x [m]',
                               yaxis_title='y [m]',
                               zaxis_title='z [m]',
                               aspectmode='manual'),
                            title_text=title)
            if filename[-5:] == '.html':
                fig.write_html(os.path.join(self.directory, title+'_'+filename))
            elif filename[-4:] == '.png':
                fig.write_image(os.path.join(self.directory, title+'_'+filename))
            else:
                raise ValueError("Only .html and .png file extensions are allowed")
            plt.close()

    def plot_3D_samples(self, model_framework, num_to_plot=100, mode='cumulativemean', filename='3D_animation.gif', axis_scale=None):
        """
        Creates an animation of 3D plots of samples.
        Parameters:
        ----------
            model_framework: dict
                Dictionary defining the model configuration. Has to have density.
            num_to_plot: int
                if mode='cumulativemean', then this is the maximum number of samples that are averaged.
                if mode='maxlikelihood', then this is the number of samples plotted in total.
            mode: str
                if 'cumulativemean': increasing number of samples are averaged up to num_to_plot
                if 'maxlikelihood': samples are arranged by decreasing likelihood and plotted
            filename: str
                has to be .gif
            axis_scale: float
                if None, then limits are inferred from voxel_grid
                else the voxel_grid is multiplied by this factor
        """
        if filename[-4:] != '.gif':
            print("Filename overwritten to .gif format")
            filename = filename[-4:]+'.gif'

        box = Box()
        box.make_voxel_grid(model_framework['grid_shape'], model_framework['ranges'])

        x = np.mean(box.voxel_grid[:,0,:], axis=1)
        y = np.mean(box.voxel_grid[:,1,:], axis=1)
        z = np.mean(box.voxel_grid[:,2,:], axis=1)

        if axis_scale is not None:
            x, y, z= x*axis_scale, y*axis_scale, z*axis_scale

        if mode == 'cumulativemean':
            models = []
            titles = []
            num = np.concatenate((np.arange(0, 100, 10), np.arange(0, num_to_plot+100, 100)[1:]
), axis=0)
            num[0] = num[0]+1
            for n in num:
                model = np.mean(self.samples[:n,:], axis=0)
                titles.append(f"Mean of {n} Random Samples")
                models.append(model)
            models = np.vstack(models)
        elif mode == 'maxlikelihood':
            indices = np.flip(np.argsort(self.log_probabilities)) #sorted by decresing probability
            models = self.samples[indices][:num_to_plot]
            titles = [f"Samples Arranged by Likelihood"]*num_to_plot
        else:
            raise ValueError('Set mode to maxlikelihood or cumulativemean.')

        image_names = []
        for i, s in enumerate(models):
            s = s - model_framework['density']
            fig = go.Figure(data=go.Volume(x=x, y=y, z=z, value=s, colorscale='Plasma', opacityscale='min', opacity=0.7, surface_count=17))
            fig.update_layout(height = 800,
                            width = 1000,
                            font = dict(size=12),
                            scene = dict(
                               xaxis_title='x [m]',
                               yaxis_title='y [m]',
                               zaxis_title='z [m]',
                               aspectmode='manual'),
                            title_text=titles[i])
            image_name = f"3D_sample_{i}.png"
            image_names.append(image_name)
            fig.write_image(os.path.join(self.directory, image_name))
            plt.close()
        
        make_gif(image_names, image_location=self.directory, filename=os.path.join(self.directory, filename))
        for im in image_names:
            os.remove(os.path.join(self.directory, im))

#    def plot_compare_voxel_slices_pygimli(self, slice_coords=[1,3,5], filename='sliced_voxels.png', plot_truth=False, normalisation=None):
#        """Makes a comparison plot consisting of slices of the voxelspace.
#        Each column is slices along a different direction (x, y, z).
#        Each row is a different slice, with increasing coordinates.
#        The method is made for 3 slices.
#        Parameters
#        ----------
#            slice_coords: list
#                The coordinate of voxels along which to slice the volume. Has to have length 3.
#            filename: str
#                The name of the file under which it will be saved.
#            plot_truth: bool
#                Defines whether the true voxelised model is added to the plot.
#            normalisation: list
#                If not None, the list has to be two elements long, and it defines the color normalisation. [minimum value of color scaler, maximum value]
#        """
#        if normalisation is not None:
#            if len(normalisation) != 2:
#                raise ValueError('The normalisation input needs to be a list with 2 elements, defining the minimum and maximum of the color scale')
#        if len(slice_coords) != 3:
#                raise ValueError('Only three slices can be defined')
#
#        if plot_truth:
#            if self.true_parameters is None:
#                raise ValueError("Give the model as the true_parameters attribute to the class")
#            true_model = self.true_parameters
#        d = round(np.power(np.shape(self.samples[0])[0], 1/3))
#        s1, s2, s3 = slice_coords
#        if plot_truth:
#            shift_idx = 0
#            plot_data = np.zeros((9, len(slice_coords)+1, d, d)) # [number of subfigures, number of subplots, dim1, dim2]
#            # Plotting the true slices
#            #true_model = np.flip(np.reshape(true_model, (d,d,d), order='F'))
#            true_model = np.flip(np.reshape(true_model, (d,d,d)))
#
#            plot_data[0, 0, :, :] = np.flip(np.rot90(true_model[s1, :, :], axes=(0,1), k=2), axis=0)
#            plot_data[3, 0, :, :] = np.flip(np.rot90(true_model[s2, :, :], axes=(0,1), k=2), axis=0)
#            plot_data[6, 0, :, :] = np.flip(np.rot90(true_model[s3, :, :], axes=(0,1), k=2), axis=0)
#
#            plot_data[1, 0, :, :] = np.flip(np.flip(true_model[:, :, s3], axis=0), axis=1)
#            plot_data[4, 0, :, :] = np.flip(np.flip(true_model[:, :, s2], axis=0), axis=1)
#            plot_data[7, 0, :, :] = np.flip(np.flip(true_model[:, :, s1], axis=0), axis=1)
#
#            plot_data[2, 0, :, :] = np.flip(np.flip(true_model[:, s3, :], axis=0), axis=1)
#            plot_data[5, 0, :, :] = np.flip(np.flip(true_model[:, s2, :], axis=0), axis=1)
#            plot_data[8, 0, :, :] = np.flip(np.flip(true_model[:, s1, :], axis=0), axis=1)
#        else:
#            shift_idx = 1
#            plot_data = np.zeros((9, len(slice_coords), d, d))
#
#        # Mean
#        mean_model = np.mean(self.samples, axis=0)
#        mean_model = np.flip(np.reshape(mean_model, (d,d,d), order='F'))
#        plot_data[0, 1-shift_idx, :, :] = np.rot90(mean_model[s1, :, :], axes=(0,1), k=3)
#        plot_data[3, 1-shift_idx, :, :] = np.rot90(mean_model[s2, :, :], axes=(0,1), k=3)
#        plot_data[6, 1-shift_idx, :, :] = np.rot90(mean_model[s3, :, :], axes=(0,1), k=3)
#
#        plot_data[1, 1-shift_idx, :, :] = np.flip(mean_model[:, s3, :], axis=1)
#        plot_data[4, 1-shift_idx, :, :] = np.flip(mean_model[:, s2, :], axis=1)
#        plot_data[7, 1-shift_idx, :, :] = np.flip(mean_model[:, s1, :], axis=1)
#
#        plot_data[2, 1-shift_idx, :, :] = np.flip(mean_model[:, :, s3], axis=1)
#        plot_data[5, 1-shift_idx, :, :] = np.flip(mean_model[:, :, s2], axis=1)
#        plot_data[8, 1-shift_idx, :, :] = np.flip(mean_model[:, :, s1], axis=1)
#
#        # Mode
#        mode_model = self.samples[np.argmax(self.log_probabilities), :]
#        mode_model = np.flip(np.reshape(mode_model, (d,d,d), order='F'))
#        plot_data[0, 2-shift_idx, :, :] = np.rot90(mode_model[s1, :, :], axes=(0,1), k=3)
#        plot_data[3, 2-shift_idx, :, :] = np.rot90(mode_model[s2, :, :], axes=(0,1), k=3)
#        plot_data[6, 2-shift_idx, :, :] = np.rot90(mode_model[s3, :, :], axes=(0,1), k=3)
#
#        plot_data[1, 2-shift_idx, :, :] = np.flip(mode_model[:, s3, :], axis=1)
#        plot_data[4, 2-shift_idx, :, :] = np.flip(mode_model[:, s2, :], axis=1)
#        plot_data[7, 2-shift_idx, :, :] = np.flip(mode_model[:, s1, :], axis=1)
#
#        plot_data[2, 2-shift_idx, :, :] = np.flip(mode_model[:, :, s3], axis=1)
#        plot_data[5, 2-shift_idx, :, :] = np.flip(mode_model[:, :, s2], axis=1)
#        plot_data[8, 2-shift_idx, :, :] = np.flip(mode_model[:, :, s1], axis=1)
#
#        # Std
#        std_model = -np.std(self.samples, axis=0)
#        std_model = np.flip(np.reshape(std_model, (d,d,d), order='F'))
#        plot_data[0, 3-shift_idx, :, :] = np.rot90(std_model[s1, :, :], axes=(0,1), k=3)
#        plot_data[3, 3-shift_idx, :, :] = np.rot90(std_model[s2, :, :], axes=(0,1), k=3)
#        plot_data[6, 3-shift_idx, :, :] = np.rot90(std_model[s3, :, :], axes=(0,1), k=3)
#
#        plot_data[1, 3-shift_idx, :, :] = np.flip(std_model[:, s3, :], axis=1)
#        plot_data[4, 3-shift_idx, :, :] = np.flip(std_model[:, s2, :], axis=1)
#        plot_data[7, 3-shift_idx, :, :] = np.flip(std_model[:, s1, :], axis=1)
#
#        plot_data[2, 3-shift_idx, :, :] = np.flip(std_model[:, :, s3], axis=1)
#        plot_data[5, 3-shift_idx, :, :] = np.flip(std_model[:, :, s2], axis=1)
#        plot_data[8, 3-shift_idx, :, :] = np.flip(std_model[:, :, s1], axis=1)
#
#        if normalisation is None:
#            norm = plt.cm.colors.Normalize(np.min(mean_model), np.max(mean_model))
#        else:
#            norm = plt.cm.colors.Normalize(normalisation[0], normalisation[1])
#
#        cmap = 'plasma'
#        fig = plt.figure(figsize=(16, 14))
#        outer = gridspec.GridSpec(3, len(slice_coords), wspace=0.2, hspace=-0.79)
#        ylabels = ['y', 'z', 'z',
#                   'y', 'z', 'z',
#                   'y', 'z', 'z']
#        xlabels = ['x', 'y', 'x',
#                   'x', 'y', 'x',
#                   'x', 'y', 'x']
#        for i in range(int(3*len(slice_coords))):
#            if plot_truth:
#                r = len(slice_coords)+1
#                inner = gridspec.GridSpecFromSubplotSpec(1, r, subplot_spec=outer[i],
#                                                     wspace=0.1, hspace=0.1)
#            else:
#                r = len(slice_coords)
#                inner = gridspec.GridSpecFromSubplotSpec(1, r, subplot_spec=outer[i],
#                                                     wspace=0.1, hspace=0.1)
#            row     = 0
#            col     = 0
#            maxCol  = 4
#
#            for j in range(r):
#                ax = plt.Subplot(fig, inner[j])
#                im = ax.imshow(plot_data[i, j, :, :], norm=norm, cmap=cmap, aspect='equal')
#                ax.set_xticks([])
#                ax.set_yticks([])
#                if i < 3:
#                    if plot_truth:
#                        if j == 0:
#                            ax.set_title('Li et al.', fontsize=12)
#                            ax.set_ylabel(ylabels[i], fontsize=12)
#                            ax.set_xlabel(xlabels[i], fontsize=12)
#                    if j == 1-shift_idx:
#                        ax.set_title("Mean", fontsize=12)
#                        if not plot_truth:
#                            ax.set_ylabel(ylabels[i], fontsize=12)
#                            ax.set_xlabel(xlabels[i], fontsize=12)
#                    if j == 2-shift_idx:
#                        ax.set_title('Mode', fontsize=12)
#                    if j == 3-shift_idx:
#                        ax.set_title('SD', fontsize=12)
#                else:
#                    if j == 0:
#                        ax.set_ylabel(ylabels[i], fontsize=12)
#                        ax.set_xlabel(xlabels[i], fontsize=12)
#                fig.add_subplot(ax)
#
#        cbar_ax = fig.add_axes([0.91, 0.35, 0.015, 0.29])
#        fig.colorbar(im, cax=cbar_ax, cmap=cmap, norm=norm)
#        cbar_ax.set_ylabel(f"\u03C1 [kg/$m^{3}$]",fontsize=12)
#        cbar_ax.tick_params(labelsize=12)
#
#        plt.savefig(os.path.join(self.directory, filename), bbox_inches='tight', transparent=True)
#        plt.close()
#
#
#
