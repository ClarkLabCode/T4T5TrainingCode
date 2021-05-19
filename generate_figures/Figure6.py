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
import matplotlib
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
matlab_data_path = os.path.join(scriptpath,'intermediate_mat_files')
output_path = os.path.join(scriptpath,'python_figure_outputs')

fig = plt.figure(dpi=150)
fig.set_size_inches(8, 5)
spec = fig.add_gridspec(nrows=4, ncols=8, height_ratios=[1,1,0.1,1],width_ratios=[1,0.2,0.7,1,0.2,0.1,1,1])

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
low_noise_color = color_cycle[0]
high_noise_color = color_cycle[1]

model_cols = [0,6,7,6,7]
data_rows = [0,1,3]
is_lnln = [False, False, False, True, True]

for model_idx, model_type in enumerate(['fly','LN_lownoise','LN_highestnoise','LNLN_lownoise','LNLN_highestnoise']):

    D = sio.loadmat(os.path.join(matlab_data_path, model_type +'.mat'))

    # Opponency
    opponency_resps = D['opponencyRespsOut'][0,0]['resps']
    if model_idx == 0:
        opponency_ind_resps = D['opponencyRespsOut'][0,0]['indResps'][0]
        opponency_err = D['opponencyRespsOut'][0,0]['sems']
    else:
        opponency_err = np.zeros(opponency_resps.shape)
    stimNames = ['PD','ND','PD+ND']
    sub_spec = spec[data_rows[0+is_lnln[model_idx]],model_cols[model_idx]].subgridspec(1,2,wspace=0.1)
    for c in range(2):
        currAxis = fig.add_subplot(sub_spec[c])
        currAxis.axhline(color='k')
        
        for stimIdx in range(3):
            alpha = 0.25 if model_idx == 0 else 1
            currAxis.bar(stimIdx,opponency_resps[stimIdx,c],yerr=opponency_err[stimIdx,c],label=stimNames[stimIdx],alpha=alpha)

        if model_idx == 0:
            data = opponency_ind_resps[c][:,0:3]
            data_x = np.tile([0,1,2],data.shape[0])
            sns.swarmplot(x=data_x.ravel(),y=data.ravel(),ax=currAxis,size=1,zorder=0)

            currAxis.set_ylim((-0.15,1.6))
            if c == 1:
                leg = currAxis.legend(bbox_to_anchor=(1,0.8))
                for lh in leg.legendHandles: 
                    lh.set_alpha(1)

        currAxis.axhline(opponency_resps[0,c],linestyle='--',color='k')        
        currAxis.axis('off')
    if model_idx:
        plt.text(-2, 10, model_type, fontsize=7)

    # Coactivation
    if not is_lnln[model_idx]:
        coactivation_resps = D['coactivationOutput'][0,0]['resps']
        currAxis = fig.add_subplot(spec[data_rows[2],model_cols[model_idx]])
        ohm = currAxis.imshow(coactivation_resps,cmap="RdBu_r",interpolation='nearest',aspect='auto',origin='upper', vmin=-1,vmax=1)
        currAxis.axis('off')

        if model_idx == 0:
            cbarAxis = fig.add_subplot(spec[3,1])
            cbar = fig.colorbar(ohm,cax=cbarAxis,boundaries=np.linspace(0,1,100),ticks=[0,0.5,1])
            cbar.ax.set_ylabel('coactivation (a.u.)')

        if model_idx:
            plt.text(0, 2, model_type, fontsize=7)


# load heatmaps
D_sparsity = sio.loadmat(os.path.join(matlab_data_path, 'sparsity_hm.mat'), squeeze_me=True)
D_opponency = sio.loadmat(os.path.join(matlab_data_path, 'opponency_hm.mat'), squeeze_me=True)
heatmap_data = [D_opponency['opponencyIndex'][:,:,:,0], D_opponency['opponencyIndex'][:,:,:,1], D_sparsity['sparsity'][:,:,:,0]]


noise_levels = [0,0.125,0.25,0.5,1]
heatmap_rows = [0,1,3]
# plot all heatmaps
cbar_labels = ['opponency','opponency','sparsity']
for i in range(3):
    currAxis = fig.add_subplot(spec[heatmap_rows[i],3])
    this_data = heatmap_data[i]
    plot_matrix = ut.convert_tensor_to_plot_matrix(this_data)
    maxabs = np.max(np.abs(plot_matrix))
    vlim = 0.36 if i == 1 else maxabs
    hm = currAxis.imshow(plot_matrix,cmap="RdBu_r", vmin= -vlim,vmax= vlim)

    currAxis.set_xticks([1,4,7,10,13])
    currAxis.set_xticks(np.arange(15)+0.5,minor=True)
    currAxis.set_xticklabels(noise_levels)

    currAxis.set_yticks([1,4,7,10,13])
    currAxis.set_yticks(np.arange(15)+0.5,minor=True)
    currAxis.set_yticklabels(noise_levels)

    currAxis.set_ylabel('training input noise')
    currAxis.set_xlabel('training output noise')

    currAxis.grid(which='minor',lw=0.25)

    currAxis.tick_params(which='both',bottom=False,left=False)

    ticks = []
    if i == 0:
        ticks = [-0.15,-0.10,-0.05,0]
        barmax = 0
        barmin = ticks[0]
    if i == 1:
        ticks = [-0.3,-0.15,0,0.15,0.3]
        barmax = 0.36
        barmin = -0.36
        outlierLocs = np.where(plot_matrix > barmax)
        for outlierIdx in range(outlierLocs[0].size):
            outlierY = outlierLocs[0][outlierIdx]+0.5
            outlierX = outlierLocs[1][outlierIdx]
            currAxis.text(outlierX, outlierY, '*', horizontalalignment='center', verticalalignment='center', fontsize=8, c=[0.8,0.8,0.8])
            print(plot_matrix[int(outlierY-0.5),int(outlierX)])
    if i == 2:
        ticks = [0,0.2,0.4,0.6,0.8]
        barmax = maxabs
        barmin = 0
    
    cbar = fig.colorbar(hm,cax=fig.add_subplot(spec[heatmap_rows[i],4]),boundaries=np.linspace(barmin,barmax,100),ticks=ticks)
    cbar.ax.set_ylabel(cbar_labels[i])

    currAxis.add_patch(matplotlib.patches.Rectangle((2.5,2.5), 3,3, fill=False, edgecolor='xkcd:bright green', lw=3,zorder=100))
    currAxis.add_patch(matplotlib.patches.Rectangle((11.5,11.5), 3,3, fill=False, edgecolor='xkcd:bright pink', lw=3,zorder=100))

fig.savefig(os.path.join(output_path,"Figure6.pdf"), bbox_inches='tight')