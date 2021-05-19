import os
import glob
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import math
import matplotlib.image as mpimg
import matplotlib.font_manager as mfm
import seaborn as sns
import pandas as pd

import utilities as ut

plt.style.use("seaborn-paper")
params = {
    'axes.labelsize': 7,
    'legend.fontsize': 6,
    'legend.framealpha': 0,
    'legend.handlelength': 1.5,
    'legend.handletextpad': 0.5,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'axes.spines.top': 'false',
    'axes.spines.right': 'false',
    'font.sans-serif': ['Arial'],
    'lines.linewidth': 1,
    'font.size': 7,
    'figure.dpi': 150
   }
plt.rcParams.update(params)

scriptpath = os.path.dirname(os.path.realpath(__file__))
parameters_path = os.path.join(scriptpath,'../training_outputs')

fig = plt.figure(dpi=150)
fig.set_size_inches(8, 3)
spec = fig.add_gridspec(nrows=1, ncols=4, width_ratios=[1,1,1,1],)

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
low_noise_color = 'xkcd:bright green'
high_noise_color = 'xkcd:bright pink'

ln_color = color_cycle[2]
lnln_color = color_cycle[3]
cond_color = color_cycle[4]

# Epoch vs Val Variance Explained

currAxis = fig.add_subplot(spec[0,0])
big_run_path = os.path.join(parameters_path,'NoiseAndModelSweep')
big_run_results = ut.load_all_params(big_run_path)
# high_noise_val_r2s = np.empty((1000,0))
# low_noise_val_r2s = np.empty((1000,0))
plot_colors = [ln_color,lnln_color,cond_color]
labels = ['LN','LNLN','Biophysical']

for model_idx,model_name in enumerate(['ln_model_flip', 'lnln_model_flip', 'conductance_model_flip']):
    val_r2s = np.empty((1000,0))
    for result_dict in big_run_results:
        if result_dict.input_noise_std == 0.125 and result_dict.output_noise_std == 0.125 and result_dict.model_name == model_name:
            val_r2s = np.column_stack((val_r2s,result_dict.val_r2))


    currAxis.plot(np.arange(1000),val_r2s,c=plot_colors[model_idx],label='high noise')
    print(np.max(val_r2s))

currAxis.set_xlabel('training epoch')
currAxis.set_ylabel('variance explained')
currAxis.set_ylim((0,0.5))

# End distributions
end_val_r2_df = pd.DataFrame(columns=['val_r2','model_name'])
for result_dict in big_run_results:
    if not(result_dict.input_noise_std == 0.125 and result_dict.output_noise_std == 0.125):
        continue

    model_name_dict = {'ln_model_flip': 'LN', 'lnln_model_flip': 'LNLN', 'conductance_model_flip': 'biophysical'}
    end_val_r2_df = end_val_r2_df.append({'variance explained': result_dict.val_r2[-1],
                                          'model': model_name_dict[result_dict.model_name]}, ignore_index=True)

currAxis = fig.add_subplot(spec[0,1])

palette_dict = {'LN': ln_color, 'LNLN': lnln_color, 'biophysical':cond_color}
sns.swarmplot(x="model", y="variance explained", hue="model", data=end_val_r2_df, palette=palette_dict, order=['LN','LNLN','biophysical'],dodge=True, size=1, ax=currAxis)
currAxis.set_ylim((0,0.5))
currAxis.axes.get_yaxis().set_visible(False)
currAxis.spines['left'].set_visible(False)

# Number of filters comparison

currAxis = fig.add_subplot(spec[0,2])
for noise_level_idx, noise_std in enumerate([0.125, 1]):
    noise_level_str = 'HighNoise' if noise_level_idx else 'LowNoise'
    filt_sweep_path = os.path.join(parameters_path,'NumFiltersSweep' + noise_level_str)
    fs_results = ut.load_all_params(filt_sweep_path)
    num_filt_options = [2, 4, 6, 8, 10]
    r2s = []
    for num_filt_idx, this_num_filt in enumerate(num_filt_options):
        r2s.append(ut.filter_and_sort_params(fs_results,num_filt = this_num_filt)[0].val_r2[-1])

    color = high_noise_color if noise_level_idx else low_noise_color
    series_label = 'high noise' if noise_level_idx else 'low noise'
    currAxis.plot(num_filt_options, r2s, c=color, marker='.', linestyle='None', label=series_label)  

currAxis.set_xlabel('# units')
currAxis.set_ylim((0,0.5))
currAxis.axes.get_yaxis().set_visible(False)
currAxis.spines['left'].set_visible(False)


# Low vs high noise eval

scriptpath = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(scriptpath,'intermediate_mat_files')
D = sio.loadmat(os.path.join(path,'varExpl.mat'),squeeze_me=True)


sub_spec = spec[0,3].subgridspec(2,2,width_ratios=[1,0.2],hspace=0.5)

fieldnames = ['varExpl','varExplNoisy']
for i in range(2):
    var_expl = D[fieldnames[i]]
    currAxis = fig.add_subplot(sub_spec[i,0])
    plot_matrix = ut.convert_tensor_to_plot_matrix(var_expl[:,:,:,0])
    plot_matrix[plot_matrix < 0] = 0
    maxabs = np.max(np.abs(plot_matrix))
    hm = currAxis.imshow(plot_matrix,cmap="RdBu_r",vmin=-maxabs,vmax=maxabs)

    noise_levels = [0,0.125,0.25,0.5,1]

    currAxis.set_xticks([1,4,7,10,13])
    currAxis.set_xticks(np.arange(15)+0.5,minor=True)
    currAxis.set_xticklabels(noise_levels)

    currAxis.set_yticks([1,4,7,10,13])
    currAxis.set_yticks(np.arange(15)+0.5,minor=True)
    currAxis.set_yticklabels(noise_levels)

    currAxis.grid(which='minor',lw=0.25)
    currAxis.tick_params(which='both',bottom=False,left=False)

    currAxis.set_ylabel('training input noise')
    if i:
        currAxis.set_xlabel('training output noise')

    ticks = []
    if i == 0:
        ticks = [0,0.1,0.2,0.3]
    if i == 1:
        ticks = [0,0.01,0.02,0.03,0.04]

    cbar = fig.colorbar(hm,cax=fig.add_subplot(sub_spec[i,-1]),boundaries=np.linspace(0,maxabs,100),ticks=ticks)
    if i == 0:
        cbar.ax.set_ylabel('variance explained')

    if i == 0:
        currAxis.set_title('evaluated at low noise')
    else:
        currAxis.set_title('evaluated at high noise')

spec.tight_layout(fig,w_pad=1.5)

# plt.show(block=True)

fig.savefig("python_figure_outputs/Figure2F_Figure4DE.pdf", bbox_inches='tight')