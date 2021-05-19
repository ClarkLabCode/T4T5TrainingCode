import os
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

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
high_contrast_color = color_cycle[0]
low_contrast_color = color_cycle[1]

scriptpath = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(scriptpath,'intermediate_mat_files')
output_path = os.path.join(scriptpath,'python_figure_outputs')

for figure_idx in range(1):

    fig = plt.figure(dpi=150)
    fig.set_size_inches(6.3, 4)

    spec = []
    model_cols = []
    data_rows = [0,2,3,4,5]
    model_names = []
    spec = fig.add_gridspec(nrows=6, ncols=6, width_ratios=[1,0.15,0.1,1,1,1],height_ratios=[1,0.8,1,1,1,1])
    model_cols = [0,3,4,5]
    model_names = ['fly','LN_nonsym','LN_tanh']
    



    for model_idx, model_type in enumerate(model_names):

        D = sio.loadmat(os.path.join(path, model_type +'.mat'))

        # Filter shape
        filters = D['filterShapesOut'][0,0]['filterShapes']
        filtMaxVal = np.max(np.abs(filters))
        filters = filters / filtMaxVal

        x = D['filterShapesOut'][0,0]['x'][0]
        t = D['filterShapesOut'][0,0]['t'][0]
        
        sub_spec = spec[data_rows[0],model_cols[model_idx]].subgridspec(1,2,wspace=0.1)
        for i in range(2):
            currAxis = fig.add_subplot(sub_spec[i])
            filt_im = currAxis.imshow(filters[:,:,i],cmap="RdBu_r",interpolation='nearest',aspect='auto',origin='upper',
                                    extent=(-7.5,7.5,t[-1],t[0]),vmin=-1,vmax=1)
            
            currAxis.set_xlabel('spatial location (deg)')
            #currAxis.set_yticklabels(['0.25'] + ['{:d}'.format(j) for j in np.round(tfs[2::2]).astype(int)])
            currAxis.set_ylabel('time in past (ms)')
            currAxis.set_xticks([-5,0,5])

            if model_idx == 0:
                if i == 0:
                    plt.title("T4")
                else:
                    plt.title('T5')
            
            if model_idx == 1:
                if i == 0:
                    plt.title("T4-like")
                else:
                    plt.title('T5-like')

            if model_idx == 4 and i == 1:
                cbar = fig.colorbar(filt_im,cax=fig.add_subplot(spec[0,-1]))
                cbar.ax.set_ylabel('filter strength (a.u.)')
                #cbar.set_ticks(np.linspace(-60,60,5))

            if model_idx == 0 and i == 0:
                currAxis.spines['bottom'].set_position(('outward', 10))
                currAxis.spines['left'].set_position(('outward', 10))
            else:
                currAxis.axis('off')

        #fig.text(0.47, 0.5, 'e', size=10, weight='bold')


        # Moving edge responses
        moving_edge_resps = D['movingEdgeRespsOut'][0,0]['resps']
        t = np.squeeze(D['movingEdgeRespsOut'][0,0]['t'])
        sub_spec = spec[data_rows[1],model_cols[model_idx]].subgridspec(1,2,wspace=0.1)
        offset_size = np.max(moving_edge_resps)/3
        t5scale = 4
        if model_idx == 0:
            moving_edge_sems = D['movingEdgeRespsOut'][0,0]['sems']
            moving_edge_resps[:,:,1] = moving_edge_resps[:,:,1] * t5scale
            moving_edge_sems[:,:,1] = moving_edge_sems[:,:,1] * t5scale
        edge_axes = [0,0]
        for i in range(2):
            currAxis = fig.add_subplot(sub_spec[i])
            edge_axes[i] = currAxis
            for edge_idx in range(4):
                if model_idx == 0:
                    p = currAxis.plot(t[:],moving_edge_resps[:,edge_idx,i] - offset_size*edge_idx)

                    bottom = moving_edge_resps[:,edge_idx,i] - moving_edge_sems[:,edge_idx,i] - offset_size*edge_idx
                    top = moving_edge_resps[:,edge_idx,i] + moving_edge_sems[:,edge_idx,i] - offset_size*edge_idx
                    currAxis.fill_between(t[:], bottom, top, alpha=0.3, facecolor=p[0].get_color())
                else:
                    currAxis.plot(t[400:700],moving_edge_resps[400:700,edge_idx,i] - offset_size*edge_idx)

            currAxis.axis('off')

            if i == 0 and model_idx == 0:
                currAxis.plot([3000, 4000], [6, 6], color='black')
                trans_offset = mtransforms.offset_copy(currAxis.transData, fig=fig,
                                                    x=0, y=0.05, units='inches')
                currAxis.text(3500, 6, '1s', horizontalalignment='center',transform=trans_offset)

                # currAxis.annotate("",xy=[-5500,-offset_size*3],xytext=[-5500,-offset_size*3 + 20],arrowprops=dict(arrowstyle="-"), annotation_clip=False)
                # trans_offset = mtransforms.offset_copy(currAxis.transData, fig=fig,
                                                    # x=0, y=0, units='inches')
                # currAxis.text(-6000, -offset_size*3, 'T4: 8 ΔF/F\nT5: 2 ΔF/F', horizontalalignment='right',transform=trans_offset)

            # if model_idx == 0:
            #     xlim = currAxis.get_xlim() # We want to disregard this bar for x lim purposes, so we'll save the current xlim and set it back later
            #     barLength = 4 if i == 0 else 4*t5scale
            #     currAxis.plot([13000, 13000], [-offset_size*3, -offset_size*3 + barLength], color='black',clip_on=False)
            #     currAxis.set_xlim(xlim)


            prop = mfm.FontProperties(family='DejaVu Sans',size=10)
            arrows = ['⇨','➡','⇦','⬅']
            if model_idx == 0 and i == 0:
                for edge_idx in range(4):
                    currAxis.text(-4500,-3 - offset_size*edge_idx, arrows[edge_idx], fontproperties=prop)

        max_ylim = max([edge_axes[0].get_ylim()[1],edge_axes[1].get_ylim()[1]])
        min_ylim = edge_axes[1].get_ylim()[0]
        edge_axes[0].set_ylim([min_ylim,max_ylim])
        edge_axes[1].set_ylim([min_ylim,max_ylim])

        # Static edges
        bars = D['staticEdgeRespsOut'][0,0]['bars']
        sub_spec = spec[data_rows[2],model_cols[model_idx]].subgridspec(2,1,hspace=0.1,height_ratios=[0.2,1])
        currAxis = fig.add_subplot(sub_spec[0])
        currAxis.imshow(bars*0.9,cmap="gray",interpolation='nearest',aspect='auto',extent=(-2.5,162.5,0,1),vmin=-1,vmax=1)
        currAxis.axis('off')

        currAxis = fig.add_subplot(sub_spec[1])
        maxResp = 0
        if model_idx > 0:
            static_edge_resps = D['staticEdgeRespsOut'][0,0]['resps']
            respLocs = D['staticEdgeRespsOut'][0,0]['respLocs'][0]
            maxResp = np.max(static_edge_resps)
            
            t4_max_moving_edge = np.max(moving_edge_resps[:,:,0])
            t5_max_moving_edge = np.max(moving_edge_resps[:,:,1])
            max_moving_edges = np.array([t4_max_moving_edge,t5_max_moving_edge])
            currAxis.plot(respLocs, static_edge_resps[:,0:2]/max_moving_edges)
            currAxis.set_ylim((0,1))
            # print(model_type)
            # print(np.max(static_edge_resps[:,0:2],0)/max_moving_edges)
        else:
            edge_resps = np.genfromtxt(os.path.join(path, 'T4T5EdgeResponses.csv'),delimiter=',',skip_header=2)
            edge_resps_reshaped = np.reshape(edge_resps,[-1,4,2])
            currAxis.plot(edge_resps_reshaped[:,0,0]+40,edge_resps_reshaped[:,0,1],label='T4')
            currAxis.plot(edge_resps_reshaped[:,1,0]+40,edge_resps_reshaped[:,1,1],label='T5')
            maxResp = 1

            currAxis.legend(bbox_to_anchor=(-0.03,1))

        currAxis.axhline(0,linestyle='--',color='k')
        currAxis.set_xlim((-2.5,162.5))

        for i in range(bars.shape[1]):
            if bars[0,i] != bars[0,i-1]:
                currAxis.axvline(i*5 -2.5,linestyle=':',color='k')

        currAxis.axis('off')

        # Opponency
        opponency_resps = D['opponencyRespsOut'][0,0]['resps']
        if model_idx == 0:
            opponency_ind_resps = D['opponencyRespsOut'][0,0]['indResps'][0]
            opponency_err = D['opponencyRespsOut'][0,0]['sems']
        else:
            opponency_err = np.zeros(opponency_resps.shape)
        stimNames = ['PD','ND','PD+ND']
        sub_spec = spec[data_rows[3],model_cols[model_idx]].subgridspec(1,2,wspace=0.1)
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
                if c == 0:
                    leg = currAxis.legend(bbox_to_anchor=(-0.05,0.8))
                    for lh in leg.legendHandles: 
                        lh.set_alpha(1)

            currAxis.axhline(opponency_resps[0,c],linestyle='--',color='k')
            
            
            
            currAxis.axis('off')

        # Coactivation
        coactivation_resps = D['coactivationOutput'][0,0]['resps']
        currAxis = fig.add_subplot(spec[data_rows[4],model_cols[model_idx]])
        ohm = currAxis.imshow(coactivation_resps,cmap="RdBu_r",interpolation='nearest',aspect='auto',origin='upper', vmin=-1,vmax=1)
        currAxis.axis('off')
        if model_idx == 0:
            cbarAxis = fig.add_subplot(spec[data_rows[4],model_cols[model_idx]+1])
            cbar = fig.colorbar(ohm,cax=cbarAxis,boundaries=np.linspace(0,1,100),ticks=[0,0.5,1])
            cbar.ax.set_ylabel('coactivation (a.u.)')

        
        for idx, name in enumerate(model_names):
            plt.text(0.2+0.18*idx, 0.8, name, fontsize=7, transform=plt.gcf().transFigure)
    #spec.tight_layout(fig,w_pad=1.5)

    fig.savefig(os.path.join(output_path, "FigureS2BC.pdf"), bbox_inches='tight')

    # plt.show(block=True)