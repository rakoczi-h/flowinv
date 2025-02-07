import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
import matplotlib.patheffects as pe

path_effects = [pe.Stroke(linewidth=2.0, foreground="black"), pe.Normal()]

location = '/data/www.astro/2263373r/giflow/4_paper/parameterised/different_data_sizes/'

directories = []
for roots, dirs, files in os.walk(location):
    for dir in dirs:
        directories.append(dirs)
directories = directories[0]
print(directories)

validation_errors = []
validation_losses = []
validation_loss_vectors = []
train_loss_vectors = []
datasizes = []
kls = []
for dir in directories:
    print(dir)
    df = pd.read_csv(os.path.join(location, dir, "loss.csv"))
    val_err = np.abs(df['val'] - df['train'])
    val_err_final = np.mean(val_err[-10:-1])
    val_loss_final = np.abs(np.mean(df['val'][-10:-1]))
    validation_errors.append(val_err_final)
    validation_losses.append(val_loss_final)
    validation_loss_vectors.append(df['val'])
    train_loss_vectors.append(df['train'])
    df = pd.read_csv(os.path.join(location, dir, "kl.csv"))
    kls.append(df['mean'])
    with open(os.path.join(location, dir, "flow_info.json"), 'rb') as f:
        info = json.load(f)
    datasize = int(info["datasize"])
    datasizes.append(datasize)


fig, ax1 = plt.subplots(figsize=(7,4))

ax2 = ax1.twinx()
ax1.grid(zorder=-1)
ax2.grid(zorder=-1)
ax1.scatter(datasizes, validation_errors, zorder=3, path_effects=path_effects, color='mediumpurple')
ax2.scatter(datasizes, kls, zorder=3, path_effects=path_effects, color='indianred')

ax1.set_xlabel('Data Size')
ax1.set_yscale('log')
ax2.set_yscale('log')
ax1.set_xscale('log')
ax1.set_ylabel('Validation Error')
ax2.set_ylabel('KL Divergence')
ax2.spines['right'].set_color('indianred')
ax2.spines['right'].set_linewidth(2)
ax2.spines['left'].set_color('mediumpurple')
ax2.spines['left'].set_linewidth(2)

ax1.tick_params(axis='y', colors='mediumpurple', which='both')
ax2.tick_params(axis='y', colors='indianred', which='both')
ax1.yaxis.label.set_color('mediumpurple')
ax2.yaxis.label.set_color('indianred')

plt.tight_layout()
plt.savefig(os.path.join(location, "valerr_dsize.png"))
plt.close()

plt.figure(figsize=(4,4))
df = pd.read_csv(os.path.join(location, 'run_2024-08-04 11:17:38.751940', "loss.csv"))
plt.grid(zorder=-1)
plt.plot(df['val'], color='mediumpurple', zorder=2, label=r'$1^6$')
plt.plot(df['train'], color='mediumpurple', alpha=0.7, zorder=2)
df = pd.read_csv(os.path.join(location, 'run_2024-08-04 11:12:41.977568', "loss.csv"))
plt.plot(df['val'], color='indianred', zorder=2, label='$1^5$')
plt.plot(df['train'], color='indianred', alpha=0.7, zorder=2)
plt.legend()
plt.ylabel('Loss')
plt.ylim(-20.0, -10.0)
plt.xlabel('Training Iterations')
plt.tight_layout()
plt.savefig(os.path.join(location, "loss_curves.png"))
plt.close()



