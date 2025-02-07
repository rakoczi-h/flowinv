
import torch
import numpy as np
import os
import pickle as pkl
import json
import pandas as pd
import matplotlib.pyplot as plt

from giflow.results import BoxFlowResults
from giflow.flowmodel import FlowModel
from giflow.plot import plot_js_hist
from giflow.read_files import read_files
from giflow.prior import Prior
import matplotlib.patheffects as pe
path_effects = [pe.Stroke(linewidth=2.0, foreground="black"), pe.Normal()]

survey_coordinates_to_include = []
model_info_to_include= []
mix_survey_order = False
num_test_cases = 10
bilby_location = '/data/www.astro/2263373r/giflow/4_paper/bilby/'
#bilby_location = None
flow_location = '/data/www.astro/2263373r/giflow/4_paper/parameterised/run_2024-07-26 13:50:53.730077/'


keys = ["px", "py", "pz", "lx", "ly", "lz", "alpha"]


js_deep_bilby = pd.read_csv(os.path.join(bilby_location, "100_testcases_deep", "js_divergence_with_prior.csv"))
js_shallow_bilby = pd.read_csv(os.path.join(bilby_location, "100_testcases_shallow", "js_divergence_with_prior.csv"))


js_deep_flow = pd.read_csv(os.path.join(flow_location, "deep_testcases", "js_divergence_with_prior.csv"))
js_shallow_flow = pd.read_csv(os.path.join(flow_location, "shallow_testcases", "js_divergence_with_prior.csv"))


jdf, jsf, jdb, jsb = [], [], [], []
for key in keys:
    jdf.append(js_deep_flow[key])
    jsf.append(js_shallow_flow[key])
    jdb.append(js_deep_bilby[key])
    jsb.append(js_shallow_bilby[key])
jdf = np.mean(np.array(jdf), axis=0)
print(np.shape(jdf))
jsf = np.mean(np.array(jsf), axis=0)
jdb = np.mean(np.array(jdb), axis=0)
jsb = np.mean(np.array(jsb), axis=0)




js_shallow = pd.read_csv(os.path.join(flow_location, "js_divergences_shallow.csv"))
js_deep = pd.read_csv(os.path.join(flow_location, "js_divergences_deep.csv"))

js_shallow_arr = []
for key in keys:
    js_shallow_arr.append(js_shallow[key])
js_shallow_arr = np.array(js_shallow_arr)
js_for_hist_shallow = js_shallow_arr.flatten()
js_mean_shallow = np.mean(js_shallow_arr, axis=0)

js_deep_arr = []
for key in keys:
    js_deep_arr.append(js_deep[key])
js_deep_arr = np.array(js_deep_arr)
js_for_hist_deep = js_deep_arr.flatten()
js_mean_deep = np.mean(js_deep_arr, axis=0)


plt.figure(figsize=(5,5))
plt.hist(js_for_hist_shallow, density=False, bins=np.logspace(np.log10(0.0001), np.log10(0.6), 20), histtype='stepfilled', alpha=0.5, range=(0.0, 0.6), label=r'$c_z > -$ 0.375 m', color='royalblue')
plt.hist(js_for_hist_shallow, density=False, bins=np.logspace(np.log10(0.0001), np.log10(0.6), 20), histtype='step', linewidth=0.7, range=(0.0, 0.6), color='k')
plt.hist(js_for_hist_deep, density=False, bins=np.logspace(np.log10(0.0001), np.log10(0.6), 20), histtype='stepfilled', alpha=0.5, range=(0.0, 0.6), label=r'$c_z < -$ 0.375 m', color='orange')
plt.hist(js_for_hist_deep, density=False, bins=np.logspace(np.log10(0.0001), np.log10(0.6), 20), histtype='step', linewidth=0.7, range=(0.0, 0.6), color='k')
plt.xscale('log')
plt.xlabel('JS Divegence', fontsize=14)
plt.legend(fontsize=11)
plt.savefig(os.path.join(flow_location, "deep_shallow_hist.png"))
plt.close()


#plot_js_hist(js_shallow_arr, keys, filename=os.path.join(flow_location, 'shallow_hist.png'))




# ------------------ JS scatter --------------------------------------
device = torch.device('cuda')
flow=FlowModel()
flow.load(flow_location)
flow.flowmodel.to(device)
flow.save_location = flow_location
data_location = flow.data_location

flow.data_location = '/scratch/balta1/2263373r/4_paper/parameterised/'
with open(os.path.join(flow.data_location, "testset_deep_0.pkl"), 'rb') as file:
    dt_deep = pkl.load(file)
with open(os.path.join(flow.data_location, "testset_shallow_0.pkl"), 'rb') as file:
    dt_shallow = pkl.load(file)

cz_deep = []
for b in dt_deep.boxes:
    cz_deep.append(b.pz)
cz_deep = np.array(cz_deep)
print(np.shape(cz_deep))

cz_shallow = []
for b in dt_shallow.boxes:
    cz_shallow.append(b.pz)
cz_shallow = np.array(cz_shallow)
print(np.shape(cz_shallow))


print(np.shape(np.c_[jdf, jdb]))
plt.figure(figsize=(5,5))
plt.vlines(cz_deep, ymin=np.min(np.c_[jdf, jdb], axis=1), ymax=np.max(np.c_[jdf, jdb], axis=1), color='black', linewidth=0.7, zorder=2)
plt.vlines(cz_shallow, ymin=np.min(np.c_[jsf, jsb], axis=1), ymax=np.max(np.c_[jsf, jsb], axis=1), color='black', linewidth=0.7, zorder=2)
plt.vlines(-0.375, ymin=-0.05, ymax=0.7, linestyle='--', linewidth=1.0, color='black')
plt.scatter(cz_deep, jdf, color='skyblue', marker='o', zorder=3, edgecolors='black', s=60)
plt.scatter(cz_deep, jdb, color='mediumpurple', marker='o', zorder=3, edgecolors='black', s=60)
plt.scatter(cz_shallow, jsf, color='moccasin', marker='o', zorder=3, edgecolors='black', s=60, label='NF')
plt.scatter(cz_shallow, jsb, color='indianred', marker='o', zorder=3, edgecolors='black', s=60, label='NS')
plt.legend(loc='lower right', fontsize=11)
plt.ylabel('JS Divergence with Prior', fontsize=14)
plt.xlabel(r'$c_z$ [m]', fontsize=14)
plt.ylim(-0.05, 0.7)
plt.grid(zorder=-1.0, linestyle="--")
plt.savefig(os.path.join(flow_location, "js_cz_scatter_2.png"))
plt.close()

#prior = dt_pp.priors
#keys = dt_pp.parameter_labels

#distributions = {"px": ['Uniform', -0.75, 0.75], "py": ['Uniform', -0.75, 0.75], "pz": ['Uniform', -0.75, 0.0],
#    "lx": ['Uniform', 0.0, 1.5], "ly": ['Uniform', 0.0, 1.5], "lz": ['Uniform', 0.0, 0.75], "alpha": ['Uniform', 0, 1.5708]}
## 0.0125 is 10% of the separation of the survey points
#prior = Prior(distributions=distributions)
#
#ppsize = 100 # THIS needs to be edited to give the overall desired data set size
#num_files = 1 #number of files that needs to be read
#pp_data, pp_conditional = read_files(data_location=data_location, filename='ppset', datasize=ppsize, num_files=num_files, survey_coordinates_to_include=survey_coordinates_to_include, model_info_to_include=model_info_to_include, mix_survey_order=mix_survey_order)
#
#pp_dataset = flow.make_tensor_dataset(pp_data, pp_conditional, device=device, scale=True)
#
#
#js_df = pd.read_csv(os.path.join(bilby_location, "100_testcases", 'js_divergence_with_prior.csv'))
#
#
#
### JS-DIVERGENCE WITH BILBY
##snrs = []
##for i in range(100):
##    snrs.append(dt_pp.surveys[i].snr())
##snrs = np.array(snrs)
##
#merit = []
#for i in range(100):
#    merit.append(dt_pp.boxes[i].pz)
#merit = np.array(merit)
##
#js_mean = []
#for key in keys:
#    js_mean.append(js_df[key])
#js_mean = np.array(js_mean)
#js_mean = np.mean(js_mean, axis=0)
##
#for key in keys:
#    plt.scatter(merit, js_df[key], label=key)
#
#plt.ylabel('js')
#plt.xlabel('pz')
#plt.savefig(os.path.join(bilby_location, "100_testcases", 'jswithprior_vs_depth.png'))
#plt.close()
##
#plt.scatter(merit, js_mean)
#plt.ylabel('js')
#plt.xlabel('pz')
#plt.savefig(os.path.join(bilby_location, "100_testcases", 'jswithprior_mean_vs_depth.png'))
#plt.close()
#
#js_100_cases = []
#for i in range(100):
##    samples, log_probabilities = flow.sample_and_logprob(pp_dataset.tensors[1][i], num=2000)
##    result_js = BoxFlowResults(samples=samples, conditional=[pp_conditional[j][i] for j in range(len(pp_conditional))], log_probabilities=log_probabilities, true_parameters=np.array([pp_data[0][i]]), parameter_labels=keys, survey_coordinates=dt_pp.surveys[0].survey_coordinates)
#    with open(os.path.join(bilby_location, "100_testcases_shallow", f"testcase_{i}", "inversion_result.json"), 'r') as file:
#        bilby_results = json.load(file)
#        bilby_posterior_dict = bilby_results['posterior']['content']
#        bilby_samples = []
#        for key in keys:
#            bilby_samples.append(bilby_posterior_dict[key])
#        bilby_samples = np.array(bilby_samples).T
#
#    js_100_cases.append(prior.get_js_divergence(bilby_samples))
#js_100_cases = np.vstack(js_100_cases).T
#data = dict.fromkeys(keys)
#for i, key in enumerate(keys):
#    data[key] = js_100_cases[i, :]
#df = pd.DataFrame(data=data)
#df.to_csv(os.path.join(bilby_location, "100_testcases_shallow", 'js_divergence_with_prior.csv'))
#
