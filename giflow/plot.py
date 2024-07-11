import pandas as pd
import scipy
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from itertools import product
from collections import namedtuple
import imageio
import os

from .box import Box
plt.style.use('seaborn-v0_8-deep')
matplotlib.rcParams['axes.titlesize'] = 10

def make_pp_plot(posterior_samples_list, truths, filename=None, confidence_interval=[0.68, 0.95, 0.997],
                 lines=None, legend_fontsize='x-small', title=True,
                 confidence_interval_alpha=0.1, fig=None, ax=None,
                 **kwargs):
    """Create a pp_plot from sets of posterior samples and their corresponding injection values.

    Parameters
    ----------
    posterior_samples_list : list
        list of posterior samples sets
    truths : list
        list of dictionaries containing the true (injected) values for each observation corresponding to `posteror_samples_list`.
    filename : str, optional
        Filename to save pp_plot in, by default None (the plot is returned) (Default: None)
    confidence_interval : list, optional
        List of shaded confidence intervals to plot, (Default: [0.68, 0.95, 0.997])
    lines : list, optional
        linestyles to use, (Default: None (a default bank of linestyles is used))
    legend_fontsize : str, optional
        legend font size descriptor, by default 'x-small'
    title : bool, optional
        Display a title with the number of observations and a combined p-value, by default True
    confidence_interval_alpha : float, optional
        Transparency of the plotted confidence interval band, by default 0.1
    fig : Figure, optional
        Existing figure to overplot the p-p plot on, by default None (a Figure is created)
    ax : Axes, optional
        Existing axes to overplot the p-p plot on, by default None (axes are created)

    Returns
    -------
    figure : Figure
        the created (or existing, if fig is not None) matplotlib Figure object
    p_values : list
        the p-value for each parameter
    """

    credible_levels = list()
    for result, truth in zip(posterior_samples_list, truths):
        credible_levels.append(get_all_credible_levels(result, truth)
        )
    credible_levels = pd.DataFrame(credible_levels)
    if lines is None:
        colors = ["C{}".format(i) for i in range(8)]
        linestyles = ["-", "--", ":", "-."]
        lines = ["{}{}".format(a, b) for a, b in product(linestyles, colors)]
    if len(lines) < len(credible_levels.keys()):
        raise ValueError("Larger number of parameters than unique linestyles")

    x_values = np.linspace(0, 1, 1001)

    N = len(credible_levels)
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(4,4))


    if isinstance(confidence_interval, float):
        confidence_interval = [confidence_interval]
    if isinstance(confidence_interval_alpha, float):
        confidence_interval_alpha = [confidence_interval_alpha] * len(confidence_interval)
    elif len(confidence_interval_alpha) != len(confidence_interval):
        raise ValueError(
            "confidence_interval_alpha must have the same length as confidence_interval")

    for ci, alpha in zip(confidence_interval, confidence_interval_alpha):
        edge_of_bound = (1. - ci) / 2.
        lower = scipy.stats.binom.ppf(1 - edge_of_bound, N, x_values) / N
        upper = scipy.stats.binom.ppf(edge_of_bound, N, x_values) / N
        # The binomial point percent function doesn't always return 0 @ 0,
        # so set those bounds explicitly to be sure
        lower[0] = 0
        upper[0] = 0
        ax.fill_between(x_values, lower, upper, alpha=alpha, color='k')

    pvalues = []
    for ii, key in enumerate(credible_levels):
        pp = np.array([sum(credible_levels[key].values < xx) /
                       len(credible_levels) for xx in x_values])
        pvalue = scipy.stats.kstest(credible_levels[key], 'uniform').pvalue
        pvalues.append(pvalue)
        try:
            name = posterior_samples_list[0].priors[key].latex_label
        except AttributeError:
            name = key
        label = "{} ({:2.3f})".format(name, pvalue)
        ax.plot(x_values, pp, lines[ii], label=label, **kwargs)
    Pvals = namedtuple('pvals', ['combined_pvalue', 'pvalues', 'names'])
    pvals = Pvals(combined_pvalue=scipy.stats.combine_pvalues(pvalues)[1],
                  pvalues=pvalues,
                  names=list(credible_levels.keys()))

    if title:
        ax.set_title("N={}, p-value={:2.4f}".format(
            len(posterior_samples_list), pvals.combined_pvalue))
    ax.set_xlabel("C.I.")
    ax.set_ylabel("Fraction of events in C.I.")
    ax.legend(handlelength=2, labelspacing=0.25, fontsize=legend_fontsize, loc='upper left')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename, dpi=500)
        plt.close()

    return fig, pvals.pvalues, pvals.combined_pvalue

def compute_credible_level(posterior_samples, truth):
    """Get the 1-d credible interval for a truth value given a set of posterior samples

    Parameters
    ----------
    posterior_samples : ndarray
        Set of posterior samples
    truth : float
        truth value to get the C.I. for

    Returns
    -------
    credible_level : float
        The C.I. value
    """
    credible_level = np.mean(np.array(posterior_samples) < truth)
    return credible_level

def get_all_credible_levels(posterior_samples, truths):
    """Get credible levels for all parameters of this event/observation, returned as a dictionary.

    Parameters
    ----------
    posterior_samples : pandas DataFrame
        A dataframe where each parameter's posterior samples has its own column.
    truths : dict
        A dictionary of the truth values for this event/observation, with the same key naming convention as `posterior_samples`.

    Returns
    -------
    dict
        The credible intervals for each parameter for this set of posterior samples.
    """

    credible_levels = {key: compute_credible_level(posterior_samples[key], truths[key]) for key in list(posterior_samples)}
    return credible_levels


def plot_js_hist(js_divs, keys, filename='js_hist.png'):
    """
    Plots a histogram of js divergences between two distributions.
    Parameters
    ----------
        js_divs: array
             An array containing the js divergence values [no. of parameters, no. of js divergence values]
        keys: list
             List of strings containing the prameter names
        filename: str
             The name under which the file is saved. (Default: 'js_hist.png')
    Outputs
    -------
        counts: list of array
            The list containing the array of counts in each bin for the seperate parameters.
        bins: array
            The array containing the edges of the bins
        median: float
            The median of the overall distribution
    """
    if filename[-4:] =! '.png':
        raise ValueError('The filetype for filename has to be .png')

    js_divs_list = []
    for i in range(np.shape(js_divs)[0]):
        js_divs_list.append(js_divs[i,:])
    js_divs_all = np.hstack(js_divs_list)
    median = np.median(js_divs_all)
    counts, bins, _ = plt.hist(js_divs_list, bins=np.logspace(np.log10(0.0001), np.log10(0.6), 20), histtype='barstacked', range=(0, 0.6), density=False, label=keys)
    plt.xscale('log')
    plt.legend()
    plt.xlabel("JS Divergence", fontsize=14)
    plt.ylabel("Counts", fontsize=14)
    plt.savefig(filename)
    plt.close()
    return counts, bins, median

def make_gif(image_names, image_location='', filename='gif.gif'):
    """
    Makes a gif out of input images.
    Parameters
    ----------
        image_names: list
            The list of the names of the image files to read.
        image_location: str
            The directory where the images are located. (Default: '')
        filename: str
            The file under which the resulting gif is saved. (Default: 'gif.gif')
    """
    if filename[-4:] =! '.gif':
        raise ValueError('The filetype for filename has to be .gif')

    images = []
    for image_name in image_names:
        images.append(imageio.imread(os.path.join(image_location, image_name)))
    imageio.mimsave(filename, images, fps=1)

# ----------------- PLOT-4-PAPER -------------------------------------
def compare_method_surveys(results_list, model_frameworks_list, survey_frameworks_list, num=1000, filename='compare_survey.png'):
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
            Number of samples to use. (Default: 1000)
        include_examples: bool
            Whether to plot a few individual samples. (Default: 'compare_survey.png')
    """
    plot_data = []
    coordinates = []
    for idx, result in enumerate(results_list):
        coordinates.append(result.survey_coordinates.copy())
        coordinates.append(result.survey_coordinates.copy())
        coordinates.append(result.survey_coordinates.copy())
        target_array = result.conditional[:,0]
        target = target_array - np.min(target_array)
        plot_data.append(target)
        mode = model_frameworks_list[idx]['type']
        gzs = []
        for i in range(num):
            if mode == 'parameterised':
                box = Box(parameterised_model=result.samples[i,:], parameter_labels=result.parameter_labels, density=model_frameworks_list[idx]['density'])
                box.translate_to_parameters()
            elif mode == 'voxelised':
                box = Box(voxelised_model=result.samples[i,:])
                box.make_voxel_grid(ranges=model_frameworks_list[idx]['ranges'], grid_shape=model_frameworks_list[idx]['grid_shape'])
            gz = box.forward_model(survey_coordinates=result.survey_coordinates.copy(), model_type=mode)-np.min(target_array)
            gzs.append(gz)
        gzs = np.array(gzs)
        mean = np.mean(gzs, axis=0)
        plot_data.append(mean)
        std = np.std(gzs, axis=0)
        plot_data.append(std)

    titles = ['Target', 'Sample Mean', 'Sample Std']
    ylabels = ['(a)', '(b)', '(c)']
    fig, axes = plt.subplots(nrows=3, ncols=3)
    plt.subplots_adjust(wspace=-0.5, hspace=0.15)
    #vmin1 = np.array([target.min(), mean.min()]).min()
    #vmin2 = std.min()
    #vmax1 = np.array([target.max(), mean.max()]).max()
    #vmax2 = std.max()
    vmin1 = 0
    vmin2 = 0
    vmax1 = 220
    vmax2 = 10
    levels1 = np.linspace(vmin1, vmax1, 256)
    levels2 = np.linspace(vmin2, vmax2, 256)
    cmap = 'rainbow'
    norm1 = matplotlib.colors.Normalize(vmin=vmin1, vmax=vmax1)
    norm2 = matplotlib.colors.Normalize(vmin=vmin2, vmax=vmax2)
    #norm = matplotlib.colors.BoundaryNorm(boundaries=levels, ncolors=15)
    for idx, ax in enumerate(axes.flatten()):
        if any([idx==i for i in [0,1,2]]):
            ax.set_title(titles[idx])
            ax.tick_params(axis='x', top=True, labeltop=True, bottom=False, labelbottom=False, labelsize=8)
        else:
            ax.tick_params(axis='x', top=False, labeltop=False, bottom=False, labelbottom=False)
        if idx==0 or idx==3 or idx==6:
            ax.plot(coordinates[idx][:,0], coordinates[idx][:,1], 'o', markersize=1, color='black')
            ax.tick_params(axis='y', left=True, labelleft=True, right=False, labelright=False, labelsize=8)
            ax.set_ylabel(ylabels[int(idx/3)], rotation=0, fontsize=10)
        else:
            ax.tick_params(axis='y', left=False, labelleft=False, right=False, labelright=False)
        if any([idx==i for i in [6,7,8]]):
            ax.set_xlabel('x[m]', fontsize=8)
        if any([idx==i for i in [2,5,8]]):
            ax.set_ylabel('y[m]', fontsize=8)
            ax.yaxis.set_label_position("right")
        if idx==2 or idx==5 or idx==8:
            ax.tricontourf(coordinates[idx][:,0], coordinates[idx][:,1], plot_data[idx], levels=levels2, cmap=cmap, norm=norm2)
            #ax.tricontour(coordinates[idx][:,0], coordinates[idx][:,1], plot_data[idx], levels=10, colors='k', linewidths=0.2)
        else:
            ax.tricontourf(coordinates[idx][:,0], coordinates[idx][:,1], plot_data[idx], levels=levels1, cmap=cmap, norm=norm1)
            #ax.tricontour(coordinates[idx][:,0], coordinates[idx][:,1], plot_data[idx], levels=10, colors='k', linewidths=0.2)
        ax.set(xlim=(np.min(coordinates[0][:,0]), np.max(coordinates[0][:,0])), ylim=(np.min(coordinates[0][:,1]), np.max(coordinates[0][:,1])), aspect='equal')
        if any([idx==i for i in [0,1,2]]):
            ax.set(title=titles[idx])
    cax1 = ax.inset_axes([-2.15, -0.3, 2.0, 0.1])
    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm1, cmap=cmap), orientation='horizontal', ticks=[0.0, 27.5, 55.0, 82.5, 110.0, 137.5, 165.0, 192.5, 220.0], boundaries=levels1, cax=cax1)
    cbar.set_label(r'$\Delta$g [$\mu$Gal]', size=8)
    cbar.ax.tick_params(rotation=45, labelsize=8)
    cax2 = ax.inset_axes([0.05, -0.3, 0.9, 0.1])
    cbar=fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm2, cmap=cmap), orientation='horizontal', ticks=[0.0, 2.5, 5.0, 7.5, 10.0], boundaries=levels2, cax=cax2)
    cbar.set_label(r'$\Delta$g [$\mu$Gal]', size=8)
    cbar.ax.tick_params(rotation=45, labelsize=8)
    plt.savefig(filename, transparent=False, bbox_inches='tight')
    plt.close()

