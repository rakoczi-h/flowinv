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

from .box import Box
from .prior import Prior
from .survey import GravitySurvey


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
        if name == 'true_parameters' or name == 'js_divergences':
            if value is not None:
                if np.shape(value)[0] != self.nparameters:
                    raise ValueError('Same number of of elements in true_parameters is required as nparameters.')
        if name == 'parameter_labels':
            if value is not None:
                if len(value) != self.nparameters:
                    raise ValueError('Same number of labels are required are nparameters.')
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

    def corner_plot(self, filename='corner.png'):
        """Makes a simple corner plot with a single set of posterior samples.
        Parameter
        ---------
        filename: str
            The name under which it is saved
        scaler: sklearn.preprocessing scaler object
            Only used if self.samples does not exist.
        """
        plot_range = []
        for dim in self.samples.T:
            plot_range.append([min(dim), max(dim)])
        if self.parameter_labels is None:
            labels = [f"q{x}" for x in range(self.nparameters)]
        else:
            labels = self.parameter_labels
        CORNER_KWARGS = dict(smooth=0.9,
                            show_titles=True,
                            label_kwargs=dict(fontsize=20),
                            title_kwargs=dict(fontsize=20),
                            quantiles=[0.16, 0.84],
                            levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
                            plot_density=False,
                            plot_datapoints=False,
                            fill_contours=True,
                            max_n_ticks=3,
                            range=plot_range,
                            labels=labels)

        figure = corner.corner(self.samples, **CORNER_KWARGS, color='#ff7f00')
        if self.true_parameters is not None:
            values = self.true_parameters
            corner.overplot_lines(figure, values, color="black")
            corner.overplot_points(figure, values[None], marker="s", color="black")
        if self.directory is not None:
            plt.savefig(os.path.join(self.directory, filename), transparent=False)
        else:
            plt.savefig(filename, transparent=False)
        plt.close()
        print("Made corner plot...")

    # fix overlaid corners method !!
    def overlaid_corner(self, other_samples, dataset_labels, filename='corner_plot_compare'):
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
        Output
        ------
        image file
        """
        _, ndim = other_samples.shape
        colors = ['#377eb8', '#ff7f00']

        n = 2
        samples_list = [self.samples, other_samples]
        max_len = max([len(s) for s in samples_list])
        plot_range = []
        for dim in range(ndim):
            plot_range.append(
                [
                    min([min(samples_list[i].T[dim]) for i in range(n)]),
                    max([max(samples_list[i].T[dim]) for i in range(n)]),
                ]
            )
        if self.parameter_labels is None:
            labels = [f"q{x}" for x in range(self.nparameters)]
        else:
            labels = self.parameter_labels

        CORNER_KWARGS = dict(
        smooth=0.9,
        show_titles=True,
        label_kwargs=dict(fontsize=20),
        title_kwargs=dict(fontsize=20),
        quantiles=[0.16, 0.84],
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
            values = self.true_parameters
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
        target_array = self.conditional[:,0]
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
        levels = np.linspace(vmin, vmax, 7)
        cmap = 'plasma'
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for idx, ax in enumerate(axes.flatten()):
            ax.plot(coordinates[:,0], coordinates[:,1], 'o', markersize=2, color='black')
            ax.tricontourf(coordinates[:,0], coordinates[:,1], plot_data[idx], levels=levels, cmap=cmap, norm=norm)
            ax.set(xlim=(np.min(coordinates[:,0]), np.max(coordinates[:,0])), ylim=(np.min(coordinates[:,1]), np.max(coordinates[:,1])), aspect='equal', title=titles[idx])
        cax = ax.inset_axes([1.1, 0.0, 0.1, 3.35])
        plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label=r'microGal')
        if self.directory is not None:
            plt.savefig(os.path.join(self.directory, filename), transparent=False)
        else:
            plt.savefig(filename, transparent=False)
        plt.close()

    def plot_compare_voxel_slices(self, slice_coords=[1,3,5], filename='sliced_voxels.png'):
        """Makes a comparison plot consisting of slices of the voxelspace.
        Each column is slices along a different direction (x, y, z).
        Each row is a different slice, with increasing coordinates.
        The method is made for 3 slices.
        Parameters
        ----------
            slice_coords: list
                The coordinate of voxels along which to slice the volume.
            filename: str
                The name of the file under which it will be saved.
        """
        if self.true_parameters is None:
            raise ValueError("Give the model as the true_parameters attribute to the class")
        true_model = self.true_parameters
        d = round(np.power(np.shape(true_model)[0], 1/3))
        s1, s2, s3 = slice_coords
        plot_data = np.zeros((9, 4, d, d)) # [number of subfigures, number of subplots, dim1, dim2]
        # Plotting the true slices
        true_model = np.reshape(true_model, (d,d,d))
        plot_data[0, 0, :, :] = true_model[s1, :, :]
        plot_data[3, 0, :, :] = true_model[s2, :, :]
        plot_data[6, 0, :, :] = true_model[s3, :, :]

        plot_data[1, 0, :, :] = true_model[:, s1, :]
        plot_data[4, 0, :, :] = true_model[:, s2, :]
        plot_data[7, 0, :, :] = true_model[:, s3, :]

        plot_data[2, 0, :, :] = true_model[:, :, s1]
        plot_data[5, 0, :, :] = true_model[:, :, s2]
        plot_data[8, 0, :, :] = true_model[:, :, s3]

        # Mean
        mean_model = np.mean(self.samples, axis=0)
        mean_model = np.reshape(mean_model, (d,d,d))
        plot_data[0, 1, :, :] = mean_model[s1, :, :]
        plot_data[3, 1, :, :] = mean_model[s2, :, :]
        plot_data[6, 1, :, :] = mean_model[s3, :, :]

        plot_data[1, 1, :, :] = mean_model[:, s1, :]
        plot_data[4, 1, :, :] = mean_model[:, s2, :]
        plot_data[7, 1, :, :] = mean_model[:, s3, :]

        plot_data[2, 1, :, :] = mean_model[:, :, s1]
        plot_data[5, 1, :, :] = mean_model[:, :, s2]
        plot_data[8, 1, :, :] = mean_model[:, :, s3]

        # Mode
        mode_model = self.samples[np.argmax(self.log_probabilities), :]
        mode_model = np.reshape(mode_model, (d,d,d))
        plot_data[0, 2, :, :] = mode_model[s1, :, :]
        plot_data[3, 2, :, :] = mode_model[s2, :, :]
        plot_data[6, 2, :, :] = mode_model[s3, :, :]

        plot_data[1, 2, :, :] = mode_model[:, s1, :]
        plot_data[4, 2, :, :] = mode_model[:, s2, :]
        plot_data[7, 2, :, :] = mode_model[:, s3, :]

        plot_data[2, 2, :, :] = mode_model[:, :, s1]
        plot_data[5, 2, :, :] = mode_model[:, :, s2]
        plot_data[8, 2, :, :] = mode_model[:, :, s3]

        # Std
        std_model = -np.std(self.samples, axis=0)
        std_model = np.reshape(std_model, (d,d,d))
        plot_data[0, 3, :, :] = std_model[s1, :, :]
        plot_data[3, 3, :, :] = std_model[s2, :, :]
        plot_data[6, 3, :, :] = std_model[s3, :, :]

        plot_data[1, 3, :, :] = std_model[:, s1, :]
        plot_data[4, 3, :, :] = std_model[:, s2, :]
        plot_data[7, 3, :, :] = std_model[:, s3, :]

        plot_data[2, 3, :, :] = std_model[:, :, s1]
        plot_data[5, 3, :, :] = std_model[:, :, s2]
        plot_data[8, 3, :, :] = std_model[:, :, s3]

        norm = plt.cm.colors.Normalize(-1500.0, 0.0)
        cmap = 'plasma'

        fig = plt.figure(figsize=(16, 14))
        outer = gridspec.GridSpec(3, 3, wspace=0.2, hspace=-0.79)
        ylabels = ['y', 'x', 'x',
                   'y', 'x', 'x',
                   'y', 'x', 'x']
        xlabels = ['z', 'z', 'y',
                   'z', 'z', 'y',
                   'z', 'z', 'y']
        for i in range(9):
            inner = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[i],
                                                     wspace=0.1, hspace=0.1)
            row     = 0
            col     = 0
            maxCol  = 4

            for j in range(4):
                ax = plt.Subplot(fig, inner[j])
                im = ax.imshow(plot_data[i, j, :, :], norm=norm, cmap=cmap, aspect='equal')
                ax.set_xticks([])
                ax.set_yticks([])
                if i < 3:
                    if j == 0:
                        ax.set_title('Target', fontsize=14)
                        ax.set_ylabel(ylabels[i], fontsize=14)
                        ax.set_xlabel(xlabels[i], fontsize=14)
                    if j == 1:
                        ax.set_title("Mean", fontsize=14)
                    if j == 2:
                        ax.set_title('Mode', fontsize=14)
                    if j == 3:
                        ax.set_title('Std', fontsize=14)
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

