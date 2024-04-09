#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 04:27:23 2023

@author: zacharysethna
"""
from __future__ import print_function, division

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pheno_def import Phenotypes

#%
class SankeyNode(object):
    """Object for sankey node"""
    
    def __init__(self, x, y, val, dx = 0.2, color = None, **kwargs):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = val
        self.x_gap = 0
        
        self.max_x = self.x + self.dx/2
        self.min_x = self.x - self.dx/2
        self.max_y = self.y + self.dy
        self.min_y = self.y
        
        self.color = color
        self.patch = mpatches.Rectangle([self.min_x+self.x_gap, self.min_y], self.dx-2*self.x_gap, self.dy, facecolor = self.color, edgecolor = 'None')
    
    def plot(self, ax):
        ax.add_patch(self.patch)
    
    def plot_node_connection(self, destination_node, ax, **kwargs):
        
        discretize = np.linspace(0, 1, 1000)
        
        x = self.max_x + (destination_node.min_x - self.max_x)*discretize
        y_shape = 1/(1+(10**2/np.power(10, 4*discretize)))
        y_shape = (y_shape - y_shape[0])/(y_shape[-1]-y_shape[0])
        
        y_top = self.max_y + (destination_node.max_y - self.max_y)*y_shape
        y_bot = self.min_y + (destination_node.min_y - self.min_y)*y_shape
        
        
        ax.fill_between(x, y_top, y_bot, fc = self.color, edgecolor = None, **kwargs)
#%
def plot_pheno_sankey(phenotypes, cell_repertoires, clones = None, **kwargs):

    fontsize = kwargs.get('fontsize', 12)
    tick_fontsize = kwargs.get('tick_fontsize', fontsize) 

    fontsize_dict = dict(xlabel_fontsize = kwargs.get('label_fontsize', fontsize),
                         ylabel_fontsize = kwargs.get('label_fontsize', fontsize),
                         title_fontsize = kwargs.get('title_fontsize', fontsize),
                         xtick_fontsize = kwargs.get('xtick_fontsize', tick_fontsize),
                         legend_fontsize = kwargs.get('legend_fontsize', fontsize)
                         )
    for kw, val in kwargs.items():
        if kw in fontsize_dict: fontsize_dict[kw] = val

    
    
    phenotypes = Phenotypes(phenotypes)
    n_reps = len(cell_repertoires)
    
    
    T10_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    colors = kwargs.get('colors', T10_colors)
    phenotype_colors = kwargs.get('phenotype_colors', {phenotype: colors[i] for i, phenotype in enumerate(phenotypes)})
    phenotype_names = kwargs.get('phenotype_names', {phenotype: phenotype for i, phenotype in enumerate(phenotypes)})
    
    normalize = kwargs.get('normalize', True)

    if 'times' in kwargs:
        times = np.array(kwargs['times'])*kwargs.get('time_rescale', 1)
        dx = (max(times) - min(times)) / 100
    else:
        times = list(range(n_reps))
    
    origin_nodes = [{} for _ in range(n_reps - 1)]
    destination_nodes = [{} for _ in range(n_reps - 1)]
    plot_nodes = [{} for _ in range(n_reps)]

    if n_reps == 1:
        i = 0
        origin_rep = cell_repertoires[0]
        c_origin_node_vals = np.zeros(len(phenotypes))

        if clones is None:
            c_clones = origin_rep.clones
        else:
            c_clones = clones
        for clone in c_clones:
            if normalize:
                origin_phenotype_vec = origin_rep.get_phenotype_counts(clone, phenotypes)/origin_rep.norm
            else:
                origin_phenotype_vec = origin_rep.get_phenotype_counts(clone, phenotypes)
            c_origin_node_vals += origin_phenotype_vec
                            
        origin_main_node_ys = np.array([0] + list(np.cumsum(c_origin_node_vals[:-1])))
        running_origin_node_ys = origin_main_node_ys.copy()
        for j, origin_phenotype in enumerate(phenotypes):
             plot_nodes[i][origin_phenotype] = SankeyNode(times[i], origin_main_node_ys[j], c_origin_node_vals[j], dx = dx, color = phenotype_colors[origin_phenotype], **kwargs)
    else:
        for i, origin_rep, dest_rep in zip(range(n_reps - 1), cell_repertoires[:-1], cell_repertoires[1:]):
            c_origin_flow_array = np.zeros((len(phenotypes), len(phenotypes)))
            c_dest_flow_array = np.zeros((len(phenotypes), len(phenotypes)))
            
            c_origin_node_vals = np.zeros(len(phenotypes))
            c_dest_node_vals = np.zeros(len(phenotypes))
            if clones is None:
                c_clones = origin_rep.clone_union(dest_rep)
            else:
                c_clones = clones
            for clone in c_clones:
                if normalize:
                    origin_phenotype_vec = origin_rep.get_phenotype_counts(clone, phenotypes)/origin_rep.norm
                    dest_phenotype_vec = dest_rep.get_phenotype_counts(clone, phenotypes)/dest_rep.norm
                else:
                    origin_phenotype_vec = origin_rep.get_phenotype_counts(clone, phenotypes)
                    dest_phenotype_vec = dest_rep.get_phenotype_counts(clone, phenotypes)
                c_origin_node_vals += origin_phenotype_vec
                c_dest_node_vals += dest_phenotype_vec
                if np.sum(origin_phenotype_vec) > 0 and np.sum(dest_phenotype_vec) > 0:
                    c_origin_flow_array += np.dot(origin_phenotype_vec.reshape(-1, 1), (dest_phenotype_vec/np.sum(dest_phenotype_vec)).reshape(1, -1))
                    c_dest_flow_array += np.dot(origin_phenotype_vec.reshape(-1, 1)/np.sum(origin_phenotype_vec), dest_phenotype_vec.reshape(1, -1))
                    
                    #c_dest_flow_array += np.dot(dest_phenotype_vec.reshape(-1, 1), (origin_phenotype_vec/np.sum(origin_phenotype_vec)).reshape(1, -1))
                    
            origin_main_node_ys = np.array([0] + list(np.cumsum(c_origin_node_vals[:-1])))
            dest_main_node_ys = np.array([0] + list(np.cumsum(c_dest_node_vals[:-1])))
            running_origin_node_ys = origin_main_node_ys.copy()
            #running_dest_node_ys = dest_main_node_ys.copy()
            #running_dest_node_ys = np.cumsum(c_dest_node_vals)
            running_dest_node_ys = np.cumsum(c_dest_node_vals) - np.sum(c_dest_flow_array, axis = 0)
            for j, origin_phenotype in enumerate(phenotypes):
                plot_nodes[i][origin_phenotype] = SankeyNode(times[i], origin_main_node_ys[j], c_origin_node_vals[j], dx = dx, color = phenotype_colors[origin_phenotype], **kwargs)
                
                for k, dest_phenotype in enumerate(phenotypes):
                    origin_nodes[i][(origin_phenotype, dest_phenotype)] = SankeyNode(times[i], running_origin_node_ys[j], c_origin_flow_array[j, k], dx = dx, color = phenotype_colors[origin_phenotype], **kwargs)
                    running_origin_node_ys[j] += c_origin_flow_array[j, k]
                    
                    # running_dest_node_ys[k] -= c_dest_flow_array[j, k]
                    # destination_nodes[i][(origin_phenotype, dest_phenotype)] = SankeyNode(times[i+1], running_dest_node_ys[k], c_dest_flow_array[j, k], **kwargs)
                    
                    destination_nodes[i][(origin_phenotype, dest_phenotype)] = SankeyNode(times[i+1], running_dest_node_ys[k], c_dest_flow_array[j, k], dx = dx, **kwargs)
                    running_dest_node_ys[k] += c_dest_flow_array[j, k]
                    
            if i == n_reps - 2:
                for j, phenotype in enumerate(phenotypes):
                    plot_nodes[i+1][phenotype] = SankeyNode(times[i+1], dest_main_node_ys[j], c_dest_node_vals[j], dx = dx, color = phenotype_colors[phenotype], **kwargs)
      
    
    if 'ax' in kwargs:
        ax = kwargs['ax']
    elif 'figsize' in kwargs:
        fig, ax = plt.subplots(figsize = kwargs['figsize'])
    else:
        fig, ax = plt.subplots(figsize = (9, 5))
        
    
    for i in range(n_reps):
        for phenotype in phenotypes:
            plot_nodes[i][phenotype].plot(ax = ax)
        if i < n_reps -1:
            for origin_phenotype in phenotypes:
                for dest_phenotype in phenotypes:
                    origin_nodes[i][(origin_phenotype, dest_phenotype)].plot_node_connection(destination_nodes[i][(origin_phenotype, dest_phenotype)], ax = ax, alpha = 0.5)
    
    if 'xlim' in kwargs:
        ax.set_xlim(kwargs['xlim'])
    if 'ylim' in kwargs:
        ax.set_ylim(kwargs['ylim'])
    else:
        if kwargs.get('normalize', True):
            ax.set_ylim([0, min(ax.get_ylim()[1], 1)])
        else:
            ax.set_ylim([0, ax.get_ylim()[1]])
        
    
    if kwargs.get('show_legend', True):
        ax.legend([plot_nodes[0][phenotype].patch for phenotype in phenotypes], [phenotype_names[phenotype] for phenotype in phenotypes], frameon = True, fontsize = fontsize_dict['legend_fontsize'], bbox_to_anchor=(1, 0.9))
    
    if kwargs.get('plot_seperate_legend', False):
        legend_fig, legend_ax = plt.subplots(figsize = (4, 3))
        legend_ax.legend([plot_nodes[0][phenotype].patch for phenotype in phenotypes], [phenotype_names[phenotype] for phenotype in phenotypes], frameon = False, fontsize = fontsize_dict['legend_fontsize'])
        legend_fig.tight_layout()
        legend_ax.set_axis_off()

    if 'names' in kwargs:
        names = kwargs['names']
    else:
        try:
            names = [cell_repertoires.name for cell_repertoires in cell_repertoires]
        except AttributeError:
            names = None
        
    if 'xlabel' in kwargs:
        ax.set_xlabel(kwargs['xlabel'], fontsize = fontsize_dict['xlabel_fontsize'])
    elif 'times' not in kwargs and names is not None:
        plt.xticks(times, names, rotation = -30, fontsize = fontsize_dict['xtick_fontsize'])
        plt.setp(ax.xaxis.get_majorticklabels(), ha="left", rotation_mode="anchor")
    if 'ylabel' in kwargs:
        ax.set_ylabel(kwargs['ylabel'], fontsize = fontsize_dict['ylabel_fontsize'])
    else:
        if kwargs.get('normalize', True):
            ax.set_ylabel('Fraction', fontsize = fontsize_dict['ylabel_fontsize'])
        else:
            ax.set_ylabel('Cell Counts', fontsize = fontsize_dict['ylabel_fontsize'])

    if 'title' in kwargs:
        ax.set_title(kwargs['title'], fontsize = fontsize_dict['title_fontsize'])

    if 'ax' not in kwargs:
        fig.tight_layout()

    if 'savefig_filename' in kwargs and 'ax' not in kwargs:
        fig.savefig(kwargs['savefig_filename'])

        
    
    
    if kwargs.get('return_axes', False):
        return fig, ax

#%%
def plot_agg_pheno_sankey_w_dict(phenotypes, cell_repertoires_by_tp, clones_by_pt = None, **kwargs):

    fontsize = kwargs.get('fontsize', 12)
    tick_fontsize = kwargs.get('tick_fontsize', fontsize) 

    fontsize_dict = dict(xlabel_fontsize = kwargs.get('label_fontsize', fontsize),
                         ylabel_fontsize = kwargs.get('label_fontsize', fontsize),
                         title_fontsize = kwargs.get('title_fontsize', fontsize),
                         xtick_fontsize = kwargs.get('xtick_fontsize', tick_fontsize),
                         legend_fontsize = kwargs.get('legend_fontsize', fontsize)
                         )
    for kw, val in kwargs.items():
        if kw in fontsize_dict: fontsize_dict[kw] = val

    
    
    phenotypes = Phenotypes(phenotypes)
    n_phases = len(cell_repertoires_by_tp)
    
    
    T10_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    colors = kwargs.get('colors', T10_colors)
    phenotype_colors = kwargs.get('phenotype_colors', {phenotype: colors[i] for i, phenotype in enumerate(phenotypes)})
    phenotype_names = kwargs.get('phenotype_names', {phenotype: phenotype for i, phenotype in enumerate(phenotypes)})
    
    
    if 'times' in kwargs:
        times = np.array(kwargs['times'])*kwargs.get('time_rescale', 1)
        
    else:
        times = list(range(n_phases))
    
    dx = (max(times) - min(times)) / 500
    
    origin_nodes = [{} for _ in range(n_phases - 1)]
    destination_nodes = [{} for _ in range(n_phases - 1)]
    plot_nodes = [{} for _ in range(n_phases)]

    if n_phases == 1:
        i = 0
        c_origin_node_vals = np.zeros(len(phenotypes))
        c_n_pts = len(cell_repertoires_by_tp[0])
        for pt, origin_rep in cell_repertoires_by_tp[0].items():
    
            if clones_by_pt is None or clones_by_pt[pt] is None:
                c_clones = origin_rep.clones
            else:
                c_clones = clones_by_pt[pt]
            for clone in c_clones:
                origin_phenotype_vec = (origin_rep.get_phenotype_counts(clone, phenotypes)/origin_rep.norm)/c_n_pts
                c_origin_node_vals += origin_phenotype_vec
                            
        origin_main_node_ys = np.array([0] + list(np.cumsum(c_origin_node_vals[:-1])))
        running_origin_node_ys = origin_main_node_ys.copy()
        for j, origin_phenotype in enumerate(phenotypes):
            plot_nodes[i][origin_phenotype] = SankeyNode(times[i], origin_main_node_ys[j], c_origin_node_vals[j], dx = dx, color = phenotype_colors[origin_phenotype], **kwargs)
    else:
        for i in range(n_phases - 1):
            print(i)
            origin_reps = cell_repertoires_by_tp[i]
            dest_reps = cell_repertoires_by_tp[i+1]
            
            c_n_origin_pts = len(origin_reps)
            c_n_dest_pts = len(dest_reps)
            
            c_origin_flow_array = np.zeros((len(phenotypes), len(phenotypes)))
            c_dest_flow_array = np.zeros((len(phenotypes), len(phenotypes)))
            
            c_origin_node_vals = np.zeros(len(phenotypes))
            c_dest_node_vals = np.zeros(len(phenotypes))
            
            for pt, origin_rep in origin_reps.items():
                if pt in dest_reps:
                    dest_rep = dest_reps[pt]
                if clones_by_pt is None or clones_by_pt[pt] is None:
                    c_clones = origin_rep.clones
                else:
                    c_clones = clones_by_pt[pt]
                for clone in c_clones:
    
                    origin_phenotype_vec = (origin_rep.get_phenotype_counts(clone, phenotypes)/origin_rep.norm)/c_n_origin_pts
                    if pt in dest_reps:
                        dest_phenotype_vec = (dest_rep.get_phenotype_counts(clone, phenotypes)/dest_rep.norm)/c_n_dest_pts
                    else:
                        dest_phenotype_vec = np.zeros(len(origin_phenotype_vec))
                    c_origin_node_vals += origin_phenotype_vec
                    c_dest_node_vals += dest_phenotype_vec
                    if np.sum(origin_phenotype_vec) > 0 and np.sum(dest_phenotype_vec) > 0:
                        c_origin_flow_array += np.dot(origin_phenotype_vec.reshape(-1, 1), (dest_phenotype_vec/np.sum(dest_phenotype_vec)).reshape(1, -1))
                        c_dest_flow_array += np.dot(origin_phenotype_vec.reshape(-1, 1)/np.sum(origin_phenotype_vec), dest_phenotype_vec.reshape(1, -1))
                        
                        #c_dest_flow_array += np.dot(dest_phenotype_vec.reshape(-1, 1), (origin_phenotype_vec/np.sum(origin_phenotype_vec)).reshape(1, -1))
                    
            origin_main_node_ys = np.array([0] + list(np.cumsum(c_origin_node_vals[:-1])))
            dest_main_node_ys = np.array([0] + list(np.cumsum(c_dest_node_vals[:-1])))
            running_origin_node_ys = origin_main_node_ys.copy()
            #running_dest_node_ys = dest_main_node_ys.copy()
            #running_dest_node_ys = np.cumsum(c_dest_node_vals)
            running_dest_node_ys = np.cumsum(c_dest_node_vals) - np.sum(c_dest_flow_array, axis = 0)
            for j, origin_phenotype in enumerate(phenotypes):
                plot_nodes[i][origin_phenotype] = SankeyNode(times[i], origin_main_node_ys[j], c_origin_node_vals[j], dx = dx, color = phenotype_colors[origin_phenotype], **kwargs)
                
                for k, dest_phenotype in enumerate(phenotypes):
                    origin_nodes[i][(origin_phenotype, dest_phenotype)] = SankeyNode(times[i], running_origin_node_ys[j], c_origin_flow_array[j, k], dx = dx, color = phenotype_colors[origin_phenotype], **kwargs)
                    running_origin_node_ys[j] += c_origin_flow_array[j, k]
                    
                    # running_dest_node_ys[k] -= c_dest_flow_array[j, k]
                    # destination_nodes[i][(origin_phenotype, dest_phenotype)] = SankeyNode(times[i+1], running_dest_node_ys[k], c_dest_flow_array[j, k], **kwargs)
                    
                    destination_nodes[i][(origin_phenotype, dest_phenotype)] = SankeyNode(times[i+1], running_dest_node_ys[k], c_dest_flow_array[j, k], dx = dx, **kwargs)
                    running_dest_node_ys[k] += c_dest_flow_array[j, k]
                    
            if i == n_phases - 2:
                for j, phenotype in enumerate(phenotypes):
                    plot_nodes[i+1][phenotype] = SankeyNode(times[i+1], dest_main_node_ys[j], c_dest_node_vals[j], dx = dx, color = phenotype_colors[phenotype], **kwargs)
      
    
    if 'ax' in kwargs:
        ax = kwargs['ax']
    elif 'figsize' in kwargs:
        fig, ax = plt.subplots(figsize = kwargs['figsize'])
    else:
        fig, ax = plt.subplots(figsize = (9, 5))
        
    
    for i in range(n_phases):
        for phenotype in phenotypes:
            plot_nodes[i][phenotype].plot(ax = ax)
        if i < n_phases -1:
            for origin_phenotype in phenotypes:
                for dest_phenotype in phenotypes:
                    origin_nodes[i][(origin_phenotype, dest_phenotype)].plot_node_connection(destination_nodes[i][(origin_phenotype, dest_phenotype)], ax = ax, alpha = 0.5)
    
    if 'xlim' in kwargs:
        ax.set_xlim(kwargs['xlim'])
    if 'ylim' in kwargs:
        ax.set_ylim(kwargs['ylim'])
    else:
        if kwargs.get('normalize', True):
            ax.set_ylim([0, min(ax.get_ylim()[1], 1)])
        else:
            ax.set_ylim([0, ax.get_ylim()[1]])
        
    
    if kwargs.get('show_legend', True):
        ax.legend([plot_nodes[0][phenotype].patch for phenotype in phenotypes], [phenotype_names[phenotype] for phenotype in phenotypes], frameon = True, fontsize = fontsize_dict['legend_fontsize'], bbox_to_anchor=(1, 0.9))
    
    if kwargs.get('plot_seperate_legend', False):
        legend_fig, legend_ax = plt.subplots(figsize = (4, 3))
        legend_ax.legend([plot_nodes[0][phenotype].patch for phenotype in phenotypes], [phenotype_names[phenotype] for phenotype in phenotypes], frameon = False, fontsize = fontsize_dict['legend_fontsize'])
        legend_fig.tight_layout()
        legend_ax.set_axis_off()

    if 'names' in kwargs:
        names = kwargs['names']
    else:
        try:
            names = [list(cell_repertoires.values())[0].name for cell_repertoires in cell_repertoires_by_tp]
        except AttributeError:
            names = None
        
    if 'xlabel' in kwargs:
        ax.set_xlabel(kwargs['xlabel'], fontsize = fontsize_dict['xlabel_fontsize'])
    elif 'times' not in kwargs and names is not None:
        plt.xticks(times, names, rotation = -30, fontsize = fontsize_dict['xtick_fontsize'])
        plt.setp(ax.xaxis.get_majorticklabels(), ha="left", rotation_mode="anchor")
    if 'ylabel' in kwargs:
        ax.set_ylabel(kwargs['ylabel'], fontsize = fontsize_dict['ylabel_fontsize'])
    else:
        if kwargs.get('normalize', True):
            ax.set_ylabel('Fraction', fontsize = fontsize_dict['ylabel_fontsize'])
        else:
            ax.set_ylabel('Cell Counts', fontsize = fontsize_dict['ylabel_fontsize'])

    if 'title' in kwargs:
        ax.set_title(kwargs['title'], fontsize = fontsize_dict['title_fontsize'])

    if 'ax' not in kwargs:
        fig.tight_layout()

    if 'savefig_filename' in kwargs and 'ax' not in kwargs:
        fig.savefig(kwargs['savefig_filename'])

        
    
    
    if kwargs.get('return_axes', False):
        return fig, ax

#%%
def setup_ternary_plot(**kwargs):
    
    fig, ax = plt.subplots(figsize = kwargs.get('figsize', (5.5, 5)))
    
    ax.axis('off')
    
    ax.plot(np.array([0, 1, 1/2, 0]), np.array([0, 0, np.sqrt(3)/2, 0]), 'k', lw = 2)
    
    # offset = 0.05
    # ax.text((-np.sqrt(3)/2)*offset, (-1/2)*offset, 'left label', ha = 'center', va = 'center', rotation = 120)
    # ax.text(1 + (np.sqrt(3)/2)*offset, (-1/2)*offset, 'right label', ha = 'center', va = 'center', rotation = 60)
    # ax.text(0.5, np.sqrt(3)/2 + offset, 'top label', ha = 'center', va = 'center')
    
    major_grid_arr = np.arange(0.2, 1, 0.2)
    minor_grid_arr = np.arange(0.1, 1, 0.1)
    
    if kwargs.get('gridlines_on', True):
        for grid_pnt in major_grid_arr:
            l_width = 0.02
            ax.plot([1-grid_pnt, (1-grid_pnt)/2], [0, np.sqrt(3)*(1-grid_pnt)/2], c = 'k', ls = ':', lw = 0.5)
            # ax.fill_between([1-grid_pnt, (1-grid_pnt+l_width)/2], 
            #                 [0, np.sqrt(3)*(1-grid_pnt-2*l_width)/2], 
            #                 [0, np.sqrt(3)*(1-grid_pnt+l_width)/2], 
            #                 color = 'k')
            # ax.fill_between([(1-grid_pnt-l_width)/2, (1-grid_pnt+l_width)/2],
            #                 [np.sqrt(3)*(1-grid_pnt-l_width)/2, np.sqrt(3)*(1-grid_pnt-2*l_width)/2],
            #                 [np.sqrt(3)*(1-grid_pnt-l_width)/2, np.sqrt(3)*(1-grid_pnt+l_width)/2],
            #                 color = 'k')
            
            
            ax.plot([grid_pnt/2, 1-grid_pnt/2], [np.sqrt(3)* grid_pnt/2, np.sqrt(3)* grid_pnt/2], c = 'k', ls = ':', lw = 0.5)
            ax.plot([grid_pnt, (1+grid_pnt)/2], [0, np.sqrt(3)*(1-grid_pnt)/2], c = 'k', ls = ':', lw = 0.5)
            
        
        
        for grid_pnt in minor_grid_arr:
            ax.plot([1-grid_pnt, (1-grid_pnt)/2], [0, np.sqrt(3)*(1-grid_pnt)/2], c = 'k', ls = ':', lw = 0.25)
            ax.plot([grid_pnt/2, 1-grid_pnt/2], [np.sqrt(3)* grid_pnt/2, np.sqrt(3)* grid_pnt/2], c = 'k', ls = ':', lw = 0.25)
            ax.plot([grid_pnt, (1+grid_pnt)/2], [0, np.sqrt(3)*(1-grid_pnt)/2], c = 'k', ls = ':', lw = 0.25)

    if kwargs.get('ticks_on', True):
        tick_fontsize = kwargs.get('tick_fontsize', 12)
        tick_len = kwargs.get('tick_length', tick_fontsize/600)
        tick_offset = kwargs.get('tick_offset', tick_fontsize/400)
        tick_width = kwargs.get('tick_width', tick_fontsize/12)
        for grid_pnt in major_grid_arr:
            
            #Ticks on left for p[1]
            # ax.plot([(1-grid_pnt)/2, (1-grid_pnt)/2 - np.sqrt(3)*tick_len/2], [np.sqrt(3)*(1-grid_pnt)/2, np.sqrt(3)*(1-grid_pnt)/2 + tick_len/2], c = 'k', lw = tick_width)
            # ax.text((1-grid_pnt)/2 - np.sqrt(3)*(tick_len+1.5*tick_offset)/2, np.sqrt(3)*(1-grid_pnt)/2 + (tick_len+tick_offset)/2, '%.1f'%(grid_pnt), ha = 'center', va = 'center', fontsize = tick_fontsize)
            ax.plot([(1-grid_pnt)/2, (1-grid_pnt)/2 - tick_len/2], [np.sqrt(3)*(1-grid_pnt)/2, np.sqrt(3)*(1-grid_pnt)/2 + np.sqrt(3)*tick_len/2], c = 'k', lw = tick_width)
            ax.text((1-grid_pnt)/2 - (tick_len+1.5*tick_offset)/2, np.sqrt(3)*(1-grid_pnt)/2 + np.sqrt(3)*(tick_len+tick_offset)/2, '%.1f'%(grid_pnt), ha = 'center', va = 'center', fontsize = tick_fontsize)

            
            #Ticks on right for p[0]
            #ax.plot([1-grid_pnt/2, 1-grid_pnt/2 + np.sqrt(3)*tick_len/2], [np.sqrt(3)* grid_pnt/2, np.sqrt(3)* grid_pnt/2 + tick_len/2], c = 'k', lw = tick_width)
            #ax.text(1-grid_pnt/2 + np.sqrt(3)*(tick_len+1.5*tick_offset)/2, np.sqrt(3)* grid_pnt/2 + (tick_len+tick_offset)/2, '%.1f'%(grid_pnt), ha = 'center', va = 'center', fontsize = tick_fontsize)
            ax.plot([1-grid_pnt/2, 1-grid_pnt/2 + tick_len], [np.sqrt(3)* grid_pnt/2, np.sqrt(3)* grid_pnt/2], c = 'k', lw = tick_width)
            ax.text(1-grid_pnt/2 + (tick_len+1.5*tick_offset), np.sqrt(3)* grid_pnt/2, '%.1f'%(grid_pnt), ha = 'center', va = 'center', fontsize = tick_fontsize)
    
            #Ticks on bottom for p[2]
            # ax.plot([grid_pnt, grid_pnt], [0, -tick_len], c = 'k', lw = tick_width)
            # ax.text(grid_pnt, -tick_len - tick_offset, '%.1f'%(grid_pnt), ha = 'center', va = 'center', fontsize = tick_fontsize)
            ax.plot([grid_pnt, grid_pnt - tick_len/2], [0, -np.sqrt(3)*tick_len/2], c = 'k', lw = tick_width)
            ax.text(grid_pnt - (tick_len+tick_offset)/2, -np.sqrt(3)*(tick_len+tick_offset)/2, '%.1f'%(grid_pnt), ha = 'center', va = 'center', fontsize = tick_fontsize)

            
    ax.set_aspect('equal', 'box')
    return fig, ax
    
def set_ternary_corner_label(ax, top_label = None, left_label = None, right_label = None, fontsize = 16, offset = 0.04):
    if top_label is not None: ax.text(0.5, np.sqrt(3)/2 + offset, top_label, ha = 'center', va = 'center', fontsize = fontsize)
    if left_label is not None: ax.text((-np.sqrt(3)/2)*offset, (-1/2)*offset, left_label, ha = 'center', va = 'center', rotation = 120, fontsize = fontsize)
    if right_label is not None: ax.text(1 + (np.sqrt(3)/2)*offset, (-1/2)*offset, right_label, ha = 'center', va = 'center', rotation = 60, fontsize = fontsize)

def set_ternary_axis_label(ax, right_label = None, left_label = None, bottom_label = None, fontsize = 16, offset = 0.1, add_arrows = True):
    if add_arrows:
        right_label = r'$\longleftarrow$' + right_label + r'$\longleftarrow$'
        left_label = r'$\longleftarrow$' + left_label + r'$\longleftarrow$'
        bottom_label = r'$\longrightarrow$' + bottom_label + r'$\longrightarrow$'

    if right_label is not None: ax.text(1-1/4 + np.sqrt(3)*(1.5*offset)/2, np.sqrt(3)/4 + offset/2, right_label, ha = 'center', va = 'center', rotation = -60, fontsize = fontsize)
    if left_label is not None: ax.text((1-1/2)/2 - np.sqrt(3)*(1.5*offset)/2, np.sqrt(3)*(1-1/2)/2 + offset/2, left_label, ha = 'center', va = 'center', rotation = 60, fontsize = fontsize)
    if bottom_label is not None: ax.text(1/2, -offset, bottom_label, ha = 'center', va = 'center', fontsize = fontsize)
    
#%
def ternary_plot_projection(p):
    return np.array([1-p[1] - p[0]/2, np.sqrt(3)*p[0]/2])
#%%
def plot_ternary_phenos(start_clones_and_phenotypes, end_clones_and_phenotypes = None, phenotypes = None, clones = None, line_type = 'arrows', c_dict = {}, s_dict = {}, kwargs_for_plots = {}, **kwargs):
    """Plots clone trajectories

    Parameters
    ----------
    clones_and_phenos : ClonesAndPhenotypes

    Returns
    -------
    None : NoneType
        If keyword return_axes == False (Default), only None is returned
    ax, cbar, fig : tuple
        If keyword return_axes == True, the axes, colorbar, and figure are
        returned. If no c_dict is provided the cbar returned will be None,
        if ax is provided as a kwargs the fig returned will be None.

    """
    
    fontsize = kwargs.get('fontsize', 16)

    fontsize_dict = {'label_fontsize': kwargs.get('label_fontsize', fontsize),
                     'title_fontsize': fontsize,
                     'cbar_label_fontsize': fontsize}
    
    for kw, val in kwargs.items():
        if kw in fontsize_dict: fontsize_dict[kw] = val

    start_marker = kwargs.get('start_marker', kwargs.get('marker', 'o'))
    end_marker = kwargs.get('end_marker', kwargs.get('marker', '^'))
    d_kwargs_for_plots = dict(
                                color = 'k',
                                size = 3,
                                alpha = 1,
                                )

    for kw, val in kwargs_for_plots.items():
        if kw == 'c':
            d_kwargs_for_plots['color'] = val
        if kw == 's':
            d_kwargs_for_plots['size'] = val
        elif kw == 'marker':
            pass
        else:
            d_kwargs_for_plots[kw] = val

    if clones is None:
        clones = start_clones_and_phenotypes.clones()
    if phenotypes is None:
        phenotypes = start_clones_and_phenotypes.phenotypes[:3]

    phenotype_names = kwargs.get('phenotype_names', {pheno: pheno for pheno in phenotypes})

    start_prob_pnts_by_clone = {c: ternary_plot_projection(np.array([start_clones_and_phenotypes[c][pheno] for pheno in phenotypes])/np.sum([start_clones_and_phenotypes[c][pheno] for pheno in phenotypes])) for c in clones if np.sum(np.array([start_clones_and_phenotypes[c][pheno] for pheno in phenotypes])) > 0}
    #Eliminate clones that cannot be plotted
    clones = [c for c in clones if c in start_prob_pnts_by_clone]
    
    color_per_clone = [c_dict.get(c, d_kwargs_for_plots['color']) for c in clones]
    size_per_clone = [s_dict.get(c, d_kwargs_for_plots['size']) for c in clones]
    d_kwargs_for_plots.__delitem__('color')
    d_kwargs_for_plots.__delitem__('size')
    if 'ax' in kwargs:
        ax = kwargs['ax']
    else:
        kwargs['figsize'] = kwargs.get('figsize', (5.45, 5))
        fig, ax = setup_ternary_plot(**kwargs)

    if end_clones_and_phenotypes is None:
        ax.scatter([start_prob_pnts_by_clone[c][0] for c in clones],[start_prob_pnts_by_clone[c][1] for c in clones], color=color_per_clone, s = [s**2 for s in size_per_clone], marker = start_marker, **d_kwargs_for_plots)
    else:
        end_prob_pnts_by_clone = {c: ternary_plot_projection(np.array([end_clones_and_phenotypes[c][pheno] for pheno in phenotypes])/np.sum([end_clones_and_phenotypes[c][pheno] for pheno in phenotypes])) for c in clones if np.sum(np.array([end_clones_and_phenotypes[c][pheno] for pheno in phenotypes])) > 0}
        for i, c in enumerate(clones):
            if c in start_prob_pnts_by_clone and c not in end_prob_pnts_by_clone:
                ax.scatter([start_prob_pnts_by_clone[c][0]], [start_prob_pnts_by_clone[c][1]], color=color_per_clone[i], s = size_per_clone[i]**2, marker = start_marker, **d_kwargs_for_plots)
            elif c not in start_prob_pnts_by_clone and c in end_prob_pnts_by_clone:
                ax.scatter([end_prob_pnts_by_clone[c][0]], [end_prob_pnts_by_clone[c][1]], color=color_per_clone[i], s = size_per_clone[i]**2, marker = end_marker, **d_kwargs_for_plots)
            elif c in start_prob_pnts_by_clone and c in end_prob_pnts_by_clone:
                if line_type == 'arrows':
                    head_length = 0.3*np.sqrt((end_prob_pnts_by_clone[c][0] - start_prob_pnts_by_clone[c][0])**2 + (end_prob_pnts_by_clone[c][1] - start_prob_pnts_by_clone[c][1])**2)
                    ax.arrow(start_prob_pnts_by_clone[c][0], start_prob_pnts_by_clone[c][1],
                              end_prob_pnts_by_clone[c][0] - start_prob_pnts_by_clone[c][0],
                              end_prob_pnts_by_clone[c][1] - start_prob_pnts_by_clone[c][1],
                              length_includes_head = True,
                              width = 0.0034*size_per_clone[i],
                              head_length = head_length,
                              facecolor=color_per_clone[i],
                              edgecolor = 'None',
                              alpha = 0.5
                             )
                    ax.arrow(start_prob_pnts_by_clone[c][0], start_prob_pnts_by_clone[c][1],
                             end_prob_pnts_by_clone[c][0] - start_prob_pnts_by_clone[c][0],
                             end_prob_pnts_by_clone[c][1] - start_prob_pnts_by_clone[c][1],
                             length_includes_head = True,
                             width = 0.0034*size_per_clone[i],
                             head_length = head_length,
                             facecolor= 'None',
                             alpha = 1,
                             edgecolor = color_per_clone[i]
                             )
                else:
                    ax.scatter([start_prob_pnts_by_clone[c][0]], [start_prob_pnts_by_clone[c][1]], color=color_per_clone[i],s = size_per_clone[i]**2, marker = start_marker, **d_kwargs_for_plots)
                    ax.scatter([end_prob_pnts_by_clone[c][0]], [end_prob_pnts_by_clone[c][1]], color=color_per_clone[i], s = size_per_clone[i]**2, marker = end_marker, **d_kwargs_for_plots)
                    ax.plot([start_prob_pnts_by_clone[c][0], end_prob_pnts_by_clone[c][0]],[start_prob_pnts_by_clone[c][1], end_prob_pnts_by_clone[c][1]], color=color_per_clone[i], lw = 0.4*size_per_clone[i], **d_kwargs_for_plots)

    set_ternary_corner_label(ax, top_label = phenotype_names[phenotypes[0]], left_label = phenotype_names[phenotypes[1]], right_label = phenotype_names[phenotypes[2]], fontsize = fontsize_dict['label_fontsize'], offset = 0.04)
    set_ternary_axis_label(ax, right_label = 'Prob(%s)'%(phenotype_names[phenotypes[0]]), left_label = 'Prob(%s)'%(phenotype_names[phenotypes[1]]), bottom_label = 'Prob(%s)'%(phenotype_names[phenotypes[2]]), fontsize = 12)
    
    
    if 'title' in kwargs:
        ax.set_title(kwargs['title'], fontsize = fontsize_dict['title_fontsize'])

    if 'ax' not in kwargs:
        fig.tight_layout()

    if 'savefig_filename' in kwargs and 'ax' not in kwargs:
        fig.savefig(kwargs['savefig_filename'])

    if kwargs.get('return_axes', False):
        if 'ax' in kwargs:
            return ax
        else:
            return fig, ax
    else:
        return None
