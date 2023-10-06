import matplotlib.pyplot as plt
import numpy as np
from .forward_models import *
from .utils import scale_data, inv_scale_data, rotate
from itertools import product
from scipy.stats import norm
import scipy.stats
import matplotlib.gridspec as gridspec
import pandas as pd
import plotly.graph_objects as go
from collections import namedtuple
import plotly
import corner
import matplotlib.lines as mlines
from plotly.subplots import make_subplots

# ------------------------------- 2D Data visualization ---------------------------
def plot_voxels(model, saveloc, filename='density_voxels'):
    """Plots the 2D projections of a 3D density distributed defined on a voxel grid.
    Parameters
    ----------
    model: array
        Contains the density values of the voxelised model.
    saveloc: str
        The location where the image is saved.
    filename: str
        The name under which the file is saved.
    Output
    ------
    image file
    """
    plt.figure(figsize=(10,7))
    norm = plt.cm.colors.Normalize(vmin=np.min(model), vmax=np.max(model))
    cmap = 'plasma'
    sc = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig, ax = plt.subplots(1, 3)
    model = model.reshape((8,8,8))
    for j in range(3):
            axi = ax[j]
            axi.imshow(np.sum(model, axis=j))
            if j == 0:
                ylabel = "y"
                xlabel = "z"
            if j == 1:
                ylabel = "x"
                xlabel = "z"
            if j == 2:
                ylabel = "x"
                xlabel = "y"
            axi.set(ylabel=ylabel, xlabel=xlabel)
    plt.savefig(saveloc+filename+'.png')
    plt.close()

def plot_survey(survey_coords, survey, saveloc, filename="survey_contour", contour=True):
    """Plots gravimetry survey data.
    Parameters
    ----------
    survey_coords: array
        Contains the xyz locations of the survey points. [no. of survey points, 3] Only the x and y are used.
    survey: array
        The gravimetry survey data.
    saveloc: str
        Where the image will be saved.
    filename: str
        The name under which the file will be saved.
    Output
    ------
    image file
    """
    n_per_side = int(np.sqrt(np.shape(survey)[0]))
    survey = np.reshape(survey, (n_per_side, n_per_side))
    survey_coords = np.reshape(survey_coords, (n_per_side, n_per_side, 3))
    plt.rc('font', size=12)
    if contour == True:
         plt.contourf(survey_coords[:,:,0], survey_coords[:,:,1], survey, cmap='plasma')
    else:
         xmin = np.min(np.min(survey_coords[:,:,0]))
         ymin = np.min(np.min(survey_coords[:,:,1]))
         xmax = np.max(np.max(survey_coords[:,:,0]))
         ymax = np.max(np.max(survey_coords[:,:,1]))
         xsep = (xmax-xmin)/(n_per_side-1)
         ysep = (ymax-ymin)/(n_per_side-1)
         plt.imshow(survey, cmap='plasma', extent=(xmin-xsep/2, xmax+xsep/2, ymin-ysep/2, ymax+ysep/2))
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.colorbar(label=f"\u0394g [\u03BCGal]")
    plt.scatter(survey_coords[:,:,0], survey_coords[:,:,1], marker='.', color='white')
    plt.savefig(saveloc+filename+'.png')
    plt.close()


# ------------------------------------ Output and target comparison --------------------------------------

def plot_compare_survey(test_samples, target_survey, survey_coords, mode='model', saveloc='', contour=True, filename='survey_compare', **kwargs):
    """Takes samples from the flow, forward models them and compares to the target survey.
    Parameters
    ----------
    test_samples: array
        The samples from the flow, containing the model parameters. [no. of samples, no. of model parameters]
    target_survey: array
        The target survey that we are inverting.
    survey coords: array
        The array containing the survey location xyz coordinates. [no. of survey points, 3]
    mode: str
        Either 'model' or 'model_parameterised'
    saveloc: str
        The location where the image is saved.
    contour: bool
         If True, a contour plot of the surveys is plotted, otherwise it is pixelated.
    filename: str
         The name of the file under which the image is saved.
    Output
    ------
    image file
    """
    len_in_1d = int(np.sqrt(np.shape(target_survey)[0]))
    target_survey = np.reshape(target_survey, (len_in_1d*len_in_1d))
    width_ratios = np.array([2.2,1,1,1])
    height_ratios = np.array([1,1])
    norm = plt.cm.colors.Normalize(vmin=np.min(target_survey), vmax=np.max(target_survey))
    cmap = 'plasma'
    fig, ax = plt.subplot_mosaic("ABCD;AEFG", width_ratios=width_ratios, height_ratios=height_ratios,
                                 gridspec_kw = {'wspace' : 0.1, 'hspace' : -0.6})
    recon_survey_diff = 0
    idx = 0
    for label, axi in ax.items():
        if idx == 0:
            plot_data = target_survey
            title = f"Target"
        else:
            if mode == 'model_parameterised':
                rho = -1600
                px = test_samples[idx-1, 0]
                py = test_samples[idx-1, 1]
                pz = test_samples[idx-1, 2]
                lx = test_samples[idx-1, 3]
                ly = test_samples[idx-1, 4]
                lz = test_samples[idx-1, 5]
                alpha_x = test_samples[idx-1, 6]
                alpha_y = test_samples[idx-1, 7]
                alpha = np.arctan(alpha_y/alpha_x)
                limits = np.array([[px-lx/2, px+lx/2], [py-ly/2, py+ly/2], [pz-lz/2, pz+lz/2]])
                survey_loc = np.zeros(np.shape(survey_coords))
                survey_loc[:,:2] = rotate(survey_coords[:,:2], origin=(px, py), angle=alpha)
                survey_loc[:,2] = survey_coords[:,2]
                plot_data = get_gz_analytical(limits, rho, survey_loc)
            elif mode == 'model':
                survey_loc = survey_coords
                print(idx-1)
                rho = test_samples[idx-1, :]
                limits = kwargs.get('voxel_grid', None)
                if limits.any() == None:
                    print("No voxel grid given")
                plot_data = get_gz_analytical_vectorised(limits, rho, survey_loc)
            else:
                raise ValueError("The mode can only be model or model_parameterised")
        recon_survey_diff = recon_survey_diff + np.mean(np.square(target_survey-plot_data))
        plot_data = np.reshape(plot_data, (len_in_1d, len_in_1d))
        x_coords = np.reshape(survey_coords[:,0], (len_in_1d, len_in_1d))
        y_coords = np.reshape(survey_coords[:,1], (len_in_1d, len_in_1d))
        if contour == True:
            im = axi.contourf(x_coords, y_coords, plot_data, cmap=cmap, norm=norm, corner_mask=False)
        else:
            im = axi.imshow(plot_data, cmap=cmap, norm=norm)
        axi.set_aspect('equal')
        if idx == 0:
            axi.scatter(x_coords, y_coords, marker='.', color='white')
            axi.set(xlabel='x [m]', ylabel='y [m]', title=f"Target", frame_on=False)
        else:
            axi.set(xticks=[], yticks=[], frame_on=False)
        idx = idx+1
    mean_rmse = np.sqrt(np.mean(recon_survey_diff))
    cbar_ax = fig.add_axes([0.92, 0.291, 0.02, 0.405]) # left, bottom, width, height
    cbar_ax.set(frame_on=False)
    fig.colorbar(im, cax=cbar_ax, cmap=cmap, norm=norm, label=f"g [\u03BCGal]")
    fig.text(0.56, 0.21, f"Mean RMSE = {mean_rmse:.3f}")
    fig.text(0.62, 0.71, "Samples", fontdict={'size' : 12})
    plt.savefig(saveloc+filename+'.png', bbox_inches='tight', transparent=True)
    plt.close()

def plot_compare_voxel_projections(output_model, target_model, saveloc, n_samples=4, filename='voxel_compare'):
    """Compares the  2D projections of the 3D outputs of the sampling method to the target distribution's projections.
    Individual samples are shown as well as the mean and standard deviation of the samples.
    Parameters
    ----------
    output_model: array
        Contains the samples from the inversion method, each representing a voxelised density model. [no. of samples, no. of parameters]
    target_model: array
        The array of density values that describe the target density distribution.
    saveloc: str
        The location where the plot is going to be saved.
    n_samples: int
        The number of samples to plot.
    filename: str
         The name of the file that the image gonna be saved as.
    Output
    ------
    image file
    """
    d = round(np.power(np.shape(target_model)[0], 1/3))
    scaling_model = np.reshape(target_model, (d,d,d))
    scaling_model = np.array([np.mean(scaling_model, axis=0), np.mean(scaling_model, axis=1), np.mean(scaling_model, axis=2)])
    norm = plt.cm.colors.Normalize(vmin=np.min(scaling_model), vmax=np.max(scaling_model))
    cmap = 'plasma'
    num_cols = n_samples+3
    fig, ax = plt.subplots(3, num_cols, gridspec_kw={'wspace' : 0.1, 'hspace' : -0.5})
    for i in range(num_cols):
        if i == 0:
            plot_model = np.reshape(target_model, (d,d,d))
            title = f"Target"
        elif i == 1:
            plot_model = np.mean(output_model, axis=0)
            plot_model = np.reshape(plot_model, (d,d,d))
            title = f"Mean"
        elif i == (num_cols-1):
            plot_model = -np.std(output_model, axis=0)
            plot_model = np.reshape(plot_model, (d,d,d))
            title = f'Std'
        else:
            plot_model = np.reshape(output_model[i, :], (d,d,d))
            title = ""
        for j in range(3):
            axi = ax[j, i]
            axi.set(frame_on=False)
            im = axi.imshow(np.mean(plot_model, axis=j), cmap=cmap, norm=norm, aspect='equal')
            if j == 0:
                axi.set_title(title, fontdict={"fontsize" :10})
            if i == 0:
                if j == 0:
                    axi.set(ylabel = "x", xlabel="z")
                elif j == 1:
                    axi.set(ylabel="y", xlabel="z")
                elif j == 2:
                    axi.set(ylabel="x", xlabel="y")
            axi.set(yticks=[], xticks=[])
    cbar_ax = fig.add_axes([0.92, 0.23, 0.02, 0.525]) # left, bottom, width, height
    cbar_ax.set(frame_on=False)
    fig.text(0.52, 0.775, "Samples", fontdict={'size' : 10})
    fig.colorbar(im, cax=cbar_ax, cmap=cmap, norm=norm, label=f"\u03C1 [kg/m$^2$]")
    plt.savefig(saveloc+filename+'.png', bbox_inches="tight", transparent=True)
    plt.close()

def plot_compare_voxel_slices(samples, logprobs, true_model, slice_coords=[1,3,5], saveloc='', filename='sliced_voxels'):
    """Makes a comparison plot consisting of slices of the voxelspace.
    Each column is slices along a different direction (x, y, z).
    Each row is a different slice, with increasing coordinates.
    The method is made for 3 slices.
    Parameters
    ----------
    samples: array
        Array containing the samples from the flow, each a voxelised density map. [no. of samples, no. of parameters]
    logprobs: array
        The log probabilties corresponding to each sample. [no. of samples]
    true_model: array
        The true density model we're comparing to. [no. of parameters]
    slice_coords: list
        The coordinate of voxels along which to slice the volume.
    saveloc: str
         The location where the image is going to be saved.
    filename: str
         The name of the file under which it will be saved.
    Output
    ------
    image file
    """
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
    mean_model = np.mean(samples, axis=0)
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
    mode_model = samples[np.argmax(logprobs), :]
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
    std_model = -np.std(samples, axis=0)
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

    norm = plt.cm.colors.Normalize(-1600.0, 0.0)
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
    cbar_ax.set_ylabel(f"\u03C1 [kg/$m^{2}$]",fontsize=14)
    cbar_ax.tick_params(labelsize=14)

    plt.savefig(saveloc+filename+'.png', bbox_inches='tight', transparent=True)
    plt.close()


# -------------------------------------- 3D Plotting --------------------------------------
# Under development
def intact_plot_3D(voxel_coords, model, saveloc='', filetype='html', filename='3d_voxelplot'):
    plotly.io.templates.default = 'plotly'
    fig = go.Figure(data=go.Volume(
        x = voxel_coords[:,0],
        y = voxel_coords[:,1],
        z = voxel_coords[:,2],
        isomax = -100,
        cmax = -500,
        cmin = -1600,
        value = model,
        opacity = 0.5,
        surface_count=25,
        opacityscale='min'
    ))
    fig.update_layout(height = 800,
                      width = 1000,
                      font = dict(size=12),
                      scene = dict(
                           xaxis_title='x [m]',
                           yaxis_title='y [m]',
                           zaxis_title='z [m]'))
    if filetype=='html':
        fig.write_html(saveloc+filename+'.html')
    if filetype=='png':
        fig.write_image(saveloc+filename+'.png')
    plt.close()

def intact_plot_3D_volume_animated(data, axis_limits, arange_by="probability", include_truth=False, saveloc='', filename='3d_mesh_plot_animated'):
    properties = data.keys()
    if arange_by not in properties:
        print("arange_by not in properties")
        exit()
    arange_idx_arr = np.argsort(data[arange_by])
    frames = []
    for i in arange_idx_arr:
        model = data['models'][i]
        voxel_coords = data['voxel_coords']
        layout_i = go.Layout(
            annotations=[
            # go.layout.Annotation(
            #         text=r'x = {px:.2f} <br>y = {py:.2f} <br>z = {pz:.2f} <br>l = {lx:.2f} <br>w = {ly:.2f} <br>d = {lz:.2f} <br>angle = {alpha:.2f} <br>probability = {probability:.2f}'.format(px=px, py=py, pz=pz, lx=lx, ly=ly, lz=lz, alpha=alpha, probability=probability),
            #         align='left',
            #         showarrow=False,
            #         xref='paper',
            #         yref='paper',
            #         x=0.175,
            #         y=-0.05,
            #         bordercolor='black',
            #         borderwidth=0),
            go.layout.Annotation(
            text=arange_by,
                    align='left',
                    showarrow=False,
                    xref='paper',
                    yref='paper',
                    x=0.5,
                    y=-0.15,
                    bordercolor='black',
                    borderwidth=0)
            ])
        frame_i = go.Frame(data=go.Volume(
            x = voxel_coords[:,0],
            y = voxel_coords[:,1],
            z = voxel_coords[:,2],
            isomax = -100,
            cmax = -500,
            cmin = -1600,
            value = model,
            opacity = 0.5,
            surface_count=25
            ),
        name=str(i),
        layout=layout_i)
        frames.append(frame_i)
    
    fig = go.Figure(frames=frames)
    fig.add_trace(go.Volume(
        x = voxel_coords[:,0],
        y = voxel_coords[:,1],
        z = voxel_coords[:,2],
        value = model,
        opacity=0.15,
        surface_count=10
        ))

    fig.add_annotation(text='text',
                    align='left',
                    showarrow=False,
                    xref='paper',
                    yref='paper',
                    x=0.8,
                    y=0.5,
                    bordercolor='black',
                    borderwidth=0)
    
    sliders = [
                {
                    "pad": {"b": 10, "t": 60},
                    "len": 0.5,
                    "x": 0.25,
                    "y": 0,
                    "steps": [
                        {
                            "args": [[f.name], frame_args(0)],
                            "label": str(data[arange_by][k]),
                            "method": "animate",
                            "name" : str(arange_by)+str(data[arange_by][k]),
                         }
                        for k, f in enumerate(fig.frames)
                    ],
                }
             ]

    # Layout
    fig.update_layout(
        scene = dict(
                           xaxis = dict(range=[axis_limits[0][0], axis_limits[0][1]],
                                        backgroundcolor="rgb(200, 200, 230)",
                                        gridcolor="white",
                                        showbackground=True,
                                        zerolinecolor="white",),
                           yaxis = dict(range=[axis_limits[1][0], axis_limits[1][1]],
                                        backgroundcolor="rgb(230, 200, 230)",
                                        gridcolor="white",
                                        showbackground=True,
                                        zerolinecolor="white",),
                           zaxis = dict(range=[axis_limits[2][0], axis_limits[2][1]],
                                        backgroundcolor="rgb(241, 236, 167)",
                                        gridcolor="white",
                                        showbackground=True,
                                        zerolinecolor="white",),
                           aspectmode='manual',
                           aspectratio=dict(x=(axis_limits[0][1]-axis_limits[0][0])/(axis_limits[0][1]-axis_limits[0][0]),
                                            y=(axis_limits[1][1]-axis_limits[1][0])/(axis_limits[0][1]-axis_limits[0][0]),
                                            z=(axis_limits[2][1]-axis_limits[2][0])/(axis_limits[0][1]-axis_limits[0][0]))),
         updatemenus = [
            {
                "buttons": [
                    {
                        "args": [None, frame_args(50)],
                        "label": "&#9654;", # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;", # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.25,
                "y": 0,
            }
         ],
         sliders=sliders
    )
    fig.write_html(saveloc+filename+'.html')
    plt.close()


def intact_plot_3D_mesh(box_params, axis_limits, filetype='html', saveloc='', filename='3D_mesh'):
    """
    box_params: needs to be 1D list or array of box parameters [px, py, pz, lx, ly, lz, alpha_x, alpha_y]
    axis_limits: list of lists, [[xmin, xmax], [ymin,ymax], [zmin,zmax]]
    """
    px = box_params[0]
    py = box_params[1]
    pz = box_params[2]
    lx = box_params[3]
    ly = box_params[4]
    lz = box_params[5]
    alpha_x = box_params[6]
    alpha_y = box_params[7]
    alpha = np.arctan(alpha_y/alpha_x)
    # define the ranges of the coordinates
    x1 = px - lx/2
    x2 = px + lx/2
    y1 = py - ly/2
    y2 = py + ly/2
    z1 = pz - lz/2
    z2 = pz + lz/2
    x_arr = np.arange(x1, x2, (x2-x1)/10) # 10 points in each grid
    y_arr = np.arange(y1, y2, (y2-y1)/10) # 10 points in each grid
    z_arr = np.arange(z1, z2, (z2-z1)/10) # 10 points in each grid
    # make grids of points for the faces
    arrs = [x_arr, y_arr, z_arr]
    lims = [[x1, x2], [y1, y2], [z1, z2]]
    points = []
    for i in range(3):
        arrs[0] = x_arr
        arrs[1] = y_arr
        arrs[2] = z_arr
        for j in range(2):
            arrs[i] = lims[i][j]
            x, y, z = np.meshgrid(arrs[0], arrs[1], arrs[2], indexing='ij')
            x, y, z = x.ravel(), y.ravel(), z.ravel()
            coords = np.c_[x, y, z]
            points.append(coords)
    points = np.vstack(points)
    points[:, :2] = rotate(points[:, :2], origin=(px, py), angle=alpha)
    # plotting
    fig = go.Figure(data=go.Mesh3d(x=points[:,0], y=points[:,1], z=points[:,2],
                    alphahull=0, # ideal fox convex bodies
                    opacity = 0.4,
                    color = "rgb(49, 5, 151)"))
    fig.update_layout(height = 800,
                      width = 1000,
                      font = dict(size=12),
                      scene = dict(
                           xaxis = dict(range=[axis_limits[0][0], axis_limits[0][1]],
                                        backgroundcolor="rgb(226, 183, 249)",
                                        gridcolor="white",
                                        showbackground=True,
                                        zerolinecolor="white",),
                           yaxis = dict(range=[axis_limits[1][0], axis_limits[1][1]],
                                        backgroundcolor="rgb(249, 173, 182)",
                                        gridcolor="white",
                                        showbackground=True,
                                        zerolinecolor="white",),
                           zaxis = dict(range=[axis_limits[2][0], axis_limits[2][1]],
                                        backgroundcolor="rgb(247, 238, 166)",
                                        gridcolor="white",
                                        showbackground=True,
                                        zerolinecolor="white",),
                           xaxis_title='x [m]',
                           yaxis_title='y [m]',
                           zaxis_title='z [m]',
                           aspectmode='manual',
                           aspectratio=dict(x=(axis_limits[0][1]-axis_limits[0][0])/(axis_limits[0][1]-axis_limits[0][0]),
                                            y=(axis_limits[1][1]-axis_limits[1][0])/(axis_limits[0][1]-axis_limits[0][0]),
                                            z=(axis_limits[2][1]-axis_limits[2][0])/(axis_limits[0][1]-axis_limits[0][0]))))
    fig.add_annotation(text=r'x = {px:.2f} m <br>y = {py:.2f} m <br>z = {pz:.2f} m <br>l = {lx:.2f} m <br>w = {ly:.2f} m <br>d = {lz:.2f} m <br>angle = {alpha:.2f} rad'.format(px=px, py=py, pz=pz, lx=lx, ly=ly, lz=lz, alpha=alpha),
                    font = dict(size=12),
                    align='left',
                    showarrow=False,
                    xref='paper',
                    yref='paper',
                    x=1.0,
                    y=0.7,
                    bordercolor='black',
                    borderwidth=1)
    if filetype=='html':
        fig.write_html(saveloc+filename+'.html')
    elif filetype=='png':
        fig.write_image(saveloc+filename+'.png')
    plt.close()


def intact_plot_3D_mesh_subplot(box_params, axis_limits, saveloc='', filename='3d_mesh_subplot'):
    fig = make_subplots(
              rows=6, cols=6,
              specs=[[{'type' : 'mesh3d'}, {'type' : 'mesh3d'}, {'type' : 'mesh3d'}, {'type' : 'mesh3d'}, {'type' : 'mesh3d'}, {'type' : 'mesh3d'}],
                     [{'type' : 'mesh3d'}, {'type' : 'mesh3d'}, {'type' : 'mesh3d'}, {'type' : 'mesh3d'}, {'type' : 'mesh3d'}, {'type' : 'mesh3d'}],
                     [{'type' : 'mesh3d'}, {'type' : 'mesh3d'}, {'type' : 'mesh3d'}, {'type' : 'mesh3d'}, {'type' : 'mesh3d'}, {'type' : 'mesh3d'}],
                     [{'type' : 'mesh3d'}, {'type' : 'mesh3d'}, {'type' : 'mesh3d'}, {'type' : 'mesh3d'}, {'type' : 'mesh3d'}, {'type' : 'mesh3d'}],
                     [{'type' : 'mesh3d'}, {'type' : 'mesh3d'}, {'type' : 'mesh3d'}, {'type' : 'mesh3d'}, {'type' : 'mesh3d'}, {'type' : 'mesh3d'}],
                     [{'type' : 'mesh3d'}, {'type' : 'mesh3d'}, {'type' : 'mesh3d'}, {'type' : 'mesh3d'}, {'type' : 'mesh3d'}, {'type' : 'mesh3d'}]])
    idx = 0
    for row in range(6):
        for col in range(6):
            px = box_params[idx, 0]
            py = box_params[idx, 1]
            pz = box_params[idx, 2]
            lx = box_params[idx, 3]
            ly = box_params[idx, 4]
            lz = box_params[idx, 5]
            alpha_x = box_params[idx, 6]
            alpha_y = box_params[idx, 7]
            alpha = np.arctan(alpha_y/alpha_x)
            # define the ranges of the coordinates
            x1 = px - lx/2
            x2 = px + lx/2
            y1 = py - ly/2
            y2 = py + ly/2
            z1 = pz - lz/2
            z2 = pz + lz/2
            x_arr = np.arange(x1, x2, (x2-x1)/10) # 10 points in each grid
            y_arr = np.arange(y1, y2, (y2-y1)/10) # 10 points in each grid
            z_arr = np.arange(z1, z2, (z2-z1)/10) # 10 points in each grid
            # make grids of points for the faces
            arrs = [x_arr, y_arr, z_arr]
            lims = [[x1, x2], [y1, y2], [z1, z2]]
            points = []
            for i in range(3):
                arrs[0] = x_arr
                arrs[1] = y_arr
                arrs[2] = z_arr
                for j in range(2):
                    arrs[i] = lims[i][j]
                    x, y, z = np.meshgrid(arrs[0], arrs[1], arrs[2], indexing='ij')
                    x, y, z = x.ravel(), y.ravel(), z.ravel()
                    coords = np.c_[x, y, z]
                    points.append(coords)
            points = np.vstack(points)
            points[:, :2] = rotate(points[:, :2], origin=(px, py), angle=alpha)
            # plotting
            fig.add_trace(go.Mesh3d(x=points[:,0], y=points[:,1], z=points[:,2],
                    alphahull=0, # ideal fox convex bodies
                    opacity = 0.4,
                    color = "rgb(49, 5, 151)"),
                    row = row+1, col=col+1)
            fig.update_layout(scene = dict(
                           xaxis = dict(range=[axis_limits[0][0], axis_limits[0][1]],
                                        backgroundcolor="rgb(226, 183, 249)",
                                        gridcolor="white",
                                        showbackground=True,
                                        zerolinecolor="white",),
                           yaxis = dict(range=[axis_limits[1][0], axis_limits[1][1]],
                                        backgroundcolor="rgb(249, 173, 182)",
                                        gridcolor="white",
                                        showbackground=True,
                                        zerolinecolor="white",),
                           zaxis = dict(range=[axis_limits[2][0], axis_limits[2][1]],
                                        backgroundcolor="rgb(247, 238, 166)",
                                        gridcolor="white",
                                        showbackground=True,
                                        zerolinecolor="white",),
                           aspectmode='manual',
                           aspectratio=dict(x=(axis_limits[0][1]-axis_limits[0][0])/(axis_limits[0][1]-axis_limits[0][0]),
                                            y=(axis_limits[1][1]-axis_limits[1][0])/(axis_limits[0][1]-axis_limits[0][0]),
                                            z=(axis_limits[2][1]-axis_limits[2][0])/(axis_limits[0][1]-axis_limits[0][0]))))
            idx =+ 1
    fig.write_html(saveloc+filename+'.html')
    plt.close()


def intact_plot_3D_mesh_animated(data, axis_limits, arange_by="probability", include_truth=False, saveloc='', filename='3d_mesh_plot_animated'):
    properties = data.keys()
    if arange_by not in properties:
        print("arange_by not in properties")
        exit()
    arange_idx_arr = np.argsort(data[arange_by])
    frames = []
    for i in arange_idx_arr:
        px = float(data['px'][i])
        py = float(data['py'][i])
        pz = float(data['pz'][i])
        lx = float(data['lx'][i])
        ly = float(data['ly'][i])
        lz = float(data['lz'][i])
        alpha_x = float(data['alpha_x'][i])
        alpha_y = float(data['alpha_y'][i])
        probability = float(data['probability'][i])
        alpha = -np.arctan(alpha_y/alpha_x)
        # define the ranges of the coordinates
        x1 = px - lx/2
        x2 = px + lx/2
        y1 = py - ly/2
        y2 = py + ly/2
        z1 = pz - lz/2
        z2 = pz + lz/2
        x_arr = np.arange(x1, x2, (x2-x1)/10) # 10 points in each grid
        y_arr = np.arange(y1, y2, (y2-y1)/10) # 10 points in each grid
        z_arr = np.arange(z1, z2, (z2-z1)/10) # 10 points in each grid
        # make grids of points for the faces
        arrs = [x_arr, y_arr, z_arr]
        lims = [[x1, x2], [y1, y2], [z1, z2]]
        points = []
        for j in range(3):
            arrs[0] = x_arr
            arrs[1] = y_arr
            arrs[2] = z_arr
            for k in range(2):
                arrs[j] = lims[j][k]
                x, y, z = np.meshgrid(arrs[0], arrs[1], arrs[2], indexing='ij')
                x, y, z = x.ravel(), y.ravel(), z.ravel()
                coords = np.c_[x, y, z]
                points.append(coords)
        points = np.vstack(points)
        points[:, :2] = rotate(points[:, :2], origin=(px, py), angle=alpha)
        # plot_arr.append(points)
        layout_i = go.Layout(
            annotations=[
            go.layout.Annotation(
                    text=r'x = {px:.2f} <br>y = {py:.2f} <br>z = {pz:.2f} <br>l = {lx:.2f} <br>w = {ly:.2f} <br>d = {lz:.2f} <br>angle = {alpha:.2f} <br>probability = {probability:.2f}'.format(px=px, py=py, pz=pz, lx=lx, ly=ly, lz=lz, alpha=alpha, probability=probability),
                    align='left',
                    showarrow=False,
                    xref='paper',
                    yref='paper',
                    x=0.175,
                    y=-0.05,
                    bordercolor='black',
                    borderwidth=0),
            go.layout.Annotation(
            text=arange_by,
                    align='left',
                    showarrow=False,
                    xref='paper',
                    yref='paper',
                    x=0.5,
                    y=-0.15,
                    bordercolor='black',
                    borderwidth=0)
            ])
        frame_i = go.Frame(data=[go.Mesh3d(x=points[:,0], y=points[:,1], z=points[:,2],
                    alphahull=0, # ideal fox convex bodies
                    opacity = 0.5,
                    color = 'darkblue')],
                    name=str(i),
                    layout=layout_i)
        frames.append(frame_i)
    
    fig = go.Figure(frames=frames)
    fig.add_trace(go.Mesh3d(x=points[:,0], y=points[:,1], z=points[:,2],
                alphahull=0, # ideal fox convex bodies
                opacity = 0.4,
                color = 'darkblue'))

    fig.add_annotation(text='text',
                    align='left',
                    showarrow=False,
                    xref='paper',
                    yref='paper',
                    x=0.8,
                    y=0.5,
                    bordercolor='black',
                    borderwidth=0)
    
    sliders = [
                {
                    "pad": {"b": 10, "t": 60},
                    "len": 0.5,
                    "x": 0.25,
                    "y": 0,
                    "steps": [
                        {
                            "args": [[f.name], frame_args(50)],
                            "label": f"{float(data[arange_by][arange_idx_arr[k]]):.2f}",
                            "method": "animate",
                            "name" : str(arange_by)+str(data[arange_by][arange_idx_arr[k]]),
                         }
                        for k, f in enumerate(fig.frames)
                    ],
                }
             ]

    # Layout
    fig.update_layout(
        scene = dict(
                           xaxis = dict(range=[axis_limits[0][0], axis_limits[0][1]],
                                        backgroundcolor="rgb(200, 200, 230)",
                                        gridcolor="white",
                                        showbackground=True,
                                        zerolinecolor="white",),
                           yaxis = dict(range=[axis_limits[1][0], axis_limits[1][1]],
                                        backgroundcolor="rgb(230, 200, 230)",
                                        gridcolor="white",
                                        showbackground=True,
                                        zerolinecolor="white",),
                           zaxis = dict(range=[axis_limits[2][0], axis_limits[2][1]],
                                        backgroundcolor="rgb(230, 230, 200)",
                                        gridcolor="white",
                                        showbackground=True,
                                        zerolinecolor="white",),
                           aspectmode='manual',
                           aspectratio=dict(x=(axis_limits[0][1]-axis_limits[0][0])/(axis_limits[0][1]-axis_limits[0][0]),
                                            y=(axis_limits[1][1]-axis_limits[1][0])/(axis_limits[0][1]-axis_limits[0][0]),
                                            z=(axis_limits[2][1]-axis_limits[2][0])/(axis_limits[0][1]-axis_limits[0][0]))),
         updatemenus = [
            {
                "buttons": [
                    {
                        "args": [None, frame_args(50)],
                        "label": "&#9654;", # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;", # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.25,
                "y": 0,
            }
         ],
         sliders=sliders
    )
    fig.write_html(saveloc+filename+'.html')
    plt.close()
    print("Made 3d plot...")

# --------------------------- Flow diagnostic plots ---------------------------------------
def plot_flow_diagnostics(latent_samples, latent_log_probs, loss, mean_kldiv, timestamp, saveloc='', filename='diagnostics'):
    """Plots diagnostics during training.
    Parameters
    ----------
    latent_samples: array
        The samples drawn from the latent space of the flow. [no. of samples, no. of parameters]
    latent_log_probs: array
         The probabilties associated with these samples. [no. of samples]
    loss: dictionary
        Contains 'train' and 'val' loss timeseries
    mean_kldiv: float
        The mean of the KL-divergence between the distribution of the latent samples and a unit gaussian
    saveloc: str
        Location where the image will be saved
    filename: str
        The name under which the image will be saved
    Output
    ------
    image file
    """
    fig, axs = plt.subplot_mosaic([['A', 'A'], ['B', 'B'], ['C', 'D']],
                              layout='constrained')
    # Plotting the loss
    ax = axs['A']
    ax.plot(loss['train'], label='Train', color='navy')
    ax.plot(loss['val'], label='Validation', color='orange')
    ax.set_xlabel('Epoch', fontdict={'fontsize': 10})
    ax.set_ylabel('Loss', fontdict={'fontsize': 10})
    ax.set_title(f"Loss | {timestamp}", fontdict={'fontsize': 10})
    ax.legend()

    # Plotting the latent space distribution
    ax = axs['B']
    for i in range(np.shape(latent_samples)[1]):
        if i == 0:
            ax.hist(latent_samples[:,i], bins=100, histtype='step', color='orange', density=True, label='Latent Samples')
        else:
            ax.hist(latent_samples[:,i], bins=100, histtype='step', color='orange', density=True)
    g = np.linspace(-4, 4, 100)
    ax.plot(g, norm.pdf(g, loc=0.0, scale=1.0), color='navy', label='Unit Gaussian')
    ax.set_xlim(-4, 4)
    ax.set_title(f"Latent Space Distribution | Mean KL = {mean_kldiv:.3f}", fontdict={'fontsize': 10})
    ax.set_xlabel('Latent Parameter', fontdict={'fontsize': 10})
    ax.set_ylabel('Sample Density', fontdict={'fontsize': 10})
    ax.legend()

    # Plotting a histogram of the latent log probabilites
    ax = axs['C']
    ax.hist(latent_log_probs, bins=100, color='navy', density=True)
    ax.set_box_aspect(1)
    ax.set_title('LS Sample Probabilities', fontdict={'fontsize': 10})
    ax.set_ylabel('Prob Density', fontdict={'fontsize': 10})
    ax.set_xlabel('Log-Prob', fontdict={'fontsize': 10})

    # Plotting an image of the correlation of the latent space samples
    ax = axs['D']
    ax.set_box_aspect(1)
    sigma = np.corrcoef(latent_samples.T)
    ax.imshow(sigma)
    ax.set_title('LS Correlation', fontdict={'fontsize': 10})

    plt.savefig(saveloc+filename+".png", transparent=True)
    plt.close()
    print("Made diagnostic plot...")

def plot_latent_hist(samples, mean_kldiv, std_kldiv, saveloc='', bins=100, filename='latent_hist'):
    for i in range(np.shape(samples)[1]):
        plt.hist(samples[:, i], bins, histtype='step', color='green', alpha=0.3, density=True)
    g = np.linspace(-4,4,100)
    plt.plot(g, norm.pdf(g, loc=0.0, scale=1.0), color='red', label='Unit gaussian')
    plt.legend()
    plt.title(f"Latent space distribution \n mean_kldiv={mean_kldiv:.3f}, std_kldiv={std_kldiv:.3f}")
    plt.xlim(-4, 4)
    plt.ylim(0, 1)
    plt.savefig(saveloc+filename+'.png')
    plt.close()

def plot_latent_logprob(log_probs, saveloc, filename='latent_logprobs.png'):
    plt.figure(figsize=(7,7))
    plt.hist(log_probs, bins=100, color='green')
    plt.title('Latent space sample log probabilities')
    plt.savefig(saveloc+filename)
    plt.close()

def plot_latent_corr(latent_samples, saveloc, filename='latent_correlation.png'):
    sigma = np.corrcoef(latent_samples.T)
    plt.figure(figsize=(7,7))
    plt.imshow(sigma)
    plt.colorbar()
    plt.savefig(saveloc+filename)
    plt.close()

def plot_loss(loss, saveloc):
    plt.figure(figsize=(7,5))
    plt.plot(loss['train'], label='Train')
    plt.plot(loss['val'], label='Val.')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(saveloc+"loss.png")
    plt.close()


# ------------------------------ Corner Plots ---------------------------------------
def corner_plot(samples, labels, values=None, saveloc='', filename='corner_plot.png'):
    """Makes a simple corner plot with a single set of posterior samples.
    Parameter
    ---------
    samples: array
        Contains the samples to plot. [no. of samples, no. of params]
    labels: list
        The parameters labels that are plotted
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
    figure = corner.corner(samples, labels=labels)
    if not values == None:
        corner.overplot_lines(figure, values, color="C1")
        corner.overplot_points(figure, values[None], marker="s", color="C1")
    plt.savefig(saveloc+filename+'.png', transparent=True)
    plt.close()
    print("Made corner plot...")

def overlaid_corner(samples_list, parameter_labels, dataset_labels, values=None, saveloc='', filename='corner_plot_compare'):
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
    n = len(samples_list)
    _, ndim = samples_list[0].shape
    max_len = max([len(s) for s in samples_list])
    colors = ['#377eb8', '#ff7f00']

    plot_range = []
    for dim in range(ndim):
        plot_range.append(
            [
                min([min(samples_list[i].T[dim]) for i in range(n)]),
                max([max(samples_list[i].T[dim]) for i in range(n)]),
            ]
        )
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
    labels=parameter_labels)

    fig = corner.corner(
        samples_list[0],
        color=colors[0],
        **CORNER_KWARGS
    )

    for idx in range(1, n):
        fig = corner.corner(
            samples_list[idx],
            fig=fig,
            weights=get_normalisation_weight(len(samples_list[idx]), max_len),
            color=colors[idx],
            **CORNER_KWARGS
        )
    if not values.any() == None:
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
    plt.savefig(saveloc+filename+'.png', transparent=True)
    plt.close()
    print("Made corner plot...")

# ------------------------------------- P-P Plotting --------------------------------------------
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
        linestyles = ["-", "--", ":"]
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

    return fig, pvals


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


# ------------------------- Plotting helper functions ------------------------------
def get_normalisation_weight(len_current_samples, len_of_longest_samples):
    return np.ones(len_current_samples) * (len_of_longest_samples / len_current_samples)

def frame_args(duration):
    return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": 0, "easing": "linear"},
        }

