import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from giflow.results import BoxFlowResults
from giflow.plot import make_gif

flow_location = '/data/www.astro/2263373r/giflow/4_paper/parameterised/initialisation_tests/'

directories = []
for roots, dirs, files in os.walk(flow_location):
    for dir in dirs:
        directories.append(dirs)
directories = directories[0]
print(directories[11])

# Corner plots
samples = []
for dir in directories:
    samples.append(pd.read_csv(os.path.join(flow_location, dir, 'testcase_2', 'samples.csv')).to_numpy()[:,1:])

parameter_labels = [r'$c_x$', r'$c_y$', r'$c_z$', r'$l_x$', r'$l_y$', r'$l_z$', r'$\alpha$']
results = BoxFlowResults(samples=samples[0], conditional=None)
results.directory = flow_location
results.overlaid_corner(other_samples=samples[1:], dataset_labels=directories, parameter_labels=parameter_labels)


image_names = []
for dir in directories:
    image_names.append(os.path.join(dir, 'testcase_2', 'corner_plot_with_prior_bounds.png'))

make_gif(image_names, image_location=flow_location, filename=os.path.join(flow_location, 'corner.gif'))



# JS histograms
js = []
for dir in directories:
    js.append(pd.read_csv(os.path.join(flow_location, dir, 'js_divergences_with_bilby.csv')).to_numpy()[:,1:])
fig, ax = plt.subplots(nrows=2, ncols=4, sharex=True, figsize=(10,5))
ax = ax.flatten()
for i in range(7):
    for j in range(20):
        ax[i].hist(js[j][:,i], bins=np.logspace(np.log10(0.0001), np.log10(0.6), 10), density=True, histtype='step')
    ax[i].set_title(parameter_labels[i])
    ax[i].set_xscale('log')
fig.delaxes(ax[7])
plt.savefig(os.path.join(flow_location, 'js_hists.png'))
plt.close()

for i in range(19):
    plt.hist(js[i][:,6], bins=10, density=True, histtype='step')
plt.savefig(os.path.join(flow_location, 'js_hists_test.png'))
plt.tight_layout()
plt.close()
