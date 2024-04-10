import pandas as pd
import scipy
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from itertools import product
from collections import namedtuple

from .box import Box
plt.style.use('seaborn-v0_8-deep')

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
        Filename to save pp_plot in, by default None (the plot is returned)
    confidence_interval : list, optional
        List of shaded confidence intervals to plot, by default [0.68, 0.95, 0.997]
    lines : list, optional
        linestyles to use, by default None (a default bank of linestyles is used)
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
    Outputs
    -------
        counts: list of array
            The list containing the array of counts in each bin for the seperate parameters.
        bins: array
            The array containing the edges of the bins
        median: float
            The median of the overall distribution
    """
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
            Number of samples to use
        include_examples: bool
            Whether to plot a few individual samples.
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

    titles = ['Target', 'Mean', 'Std']
    fig, axes = plt.subplots(nrows=3, ncols=3)
    vmin = np.array([target.min(), mean.min()]).min()
    #vmax = np.array([target.max(), mean.max()]).max()
    vmax = 210
    levels = np.linspace(vmin, vmax, 15)
    cmap = 'plasma'
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    #norm = matplotlib.colors.BoundaryNorm(boundaries=levels, ncolors=15)
    for idx, ax in enumerate(axes.flatten()):
        if idx==0 or idx==3 or idx==6:
            ax.plot(coordinates[idx][:,0], coordinates[idx][:,1], 'o', markersize=2, color='black')
        ax.tricontourf(coordinates[idx][:,0], coordinates[idx][:,1], plot_data[idx], levels=levels, cmap=cmap, norm=norm)
        ax.set(xlim=(np.min(coordinates[idx][:,0]), np.max(coordinates[idx][:,0])), ylim=(np.min(coordinates[idx][:,1]), np.max(coordinates[idx][:,1])), aspect='equal')
        if any([idx==i for i in [0,1,2]]):
            ax.set(title=titles[idx])
    cax = ax.inset_axes([1.1, 0.0, 0.1, 3.35])
    plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=levels, boundaries=levels, cax=cax, label=r'microGal')
    plt.savefig(filename, transparent=False)
    plt.close()

