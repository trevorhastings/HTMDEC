# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 11:56:00 2025

@author: Timet
"""

def change_directory_to_current_py_file():
    try:
        import os
        os.chdir(os.path.split(os.path.abspath(__file__))[0])
    except Exception as e:
        print(f'\nDirectory change or os module error:\n\n{e}')

change_directory_to_current_py_file()

from package_BBO_functions import import_excel
from package_BBO_functions import pareto_maximum, hypervolume, hypervolume_smallest

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
import pandas as pd
import numpy as np

folder_name = 'Test simulations/graph example 1'
file_name = f'{folder_name}/inputSpace_from1000_size20_iter6'

# folder_name = 'Test simulations/graph example 2'
# file_name = f'{folder_name}/inputSpace_from1000_size20_iter10'

data_input = import_excel(f'{file_name}.xlsx')
batch_size = int(data_input.shape[0] / data_input['Iteration'].max())

file_name_space = 'compositionSpace_htmdec_1000_withtype'

phase_space = import_excel(f'{file_name_space}.xlsx')
outputs = ['PROP RT Density (g/cm3)','PROP RT Heat Cap (J/(mol K))']

[pareto_maximum(data_input.iloc[index_start:(index_start + batch_size)][outputs].to_numpy()).shape[0] for index_start in range(0, data_input.shape[0], batch_size)]

pareto_set = pareto_maximum(phase_space[outputs].to_numpy())
pareto_set_full_data = pd.merge(pd.DataFrame(pareto_set, columns = outputs), phase_space, on = outputs, how = 'left')
pareto_set_full_data['Pareto'] = 'y'
phase_space_with_pareto = phase_space.copy()
phase_space_with_pareto = pd.merge(phase_space_with_pareto, pareto_set_full_data, how = 'outer')
phase_space_with_pareto = phase_space_with_pareto.fillna('n')

phase_space_hypervolume_min = hypervolume_smallest(
    dataset = phase_space[outputs].to_numpy(), num_datapoints = batch_size)
phase_space_hypervolume = hypervolume(pareto_set)

nondominated = data_input[data_input['Pareto'] == 'y']

data_min_x = min(phase_space_with_pareto[outputs[0]])
data_max_x = max(phase_space_with_pareto[outputs[0]])
data_min_y = min(phase_space_with_pareto[outputs[1]])
data_max_y = max(phase_space_with_pareto[outputs[1]])
xrange = data_max_x - data_min_x
yrange = data_max_y - data_min_y

fig, ax = plt.subplots(figsize=(6, 6), frameon=True)

ax.set_xlabel(outputs[0], fontsize=24)
ax.set_ylabel(outputs[1], fontsize=24)
ax.tick_params(axis='both', which='major', labelsize=24)
# ax.set_xticks([i for i in range(int(round(data_min_x)), int(round(data_max_x + 1)) + 1, 1)])
# ax.set_yticks([i for i in range(int(round(data_min_y)), int(round(data_max_y + 1)) + 1, 1)])
# ax.set_xticklabels([i for i in range(int(round(data_min_x)), int(round(data_max_x + 1)) + 1, 1)], fontsize=24)
# ax.set_yticklabels([i for i in range(int(round(data_min_y)), int(round(data_max_y + 1)) + 1, 1)], fontsize=24)

ax_main_plot_range_y = data_max_y - data_min_y
ax_main_plot_ymin = data_min_y - 0.2 * ax_main_plot_range_y
ax_main_plot_ymax = data_max_y + 0.3 * ax_main_plot_range_y
ax_main_plot_range_x = data_max_x - data_min_x
ax_main_plot_xmin = data_min_x - 0.1 * ax_main_plot_range_x
ax_main_plot_xmax = data_max_x + 0.15 * ax_main_plot_range_x

ax.set_xlim(ax_main_plot_xmin, ax_main_plot_xmax)
ax.set_ylim(ax_main_plot_ymin, ax_main_plot_ymax)

ax.scatter(
    data_input[outputs[0]], data_input[outputs[1]],
    c=data_input['Iteration'], cmap='viridis',
    edgecolor='k', linewidths=0.8, s=150)
ax.scatter(
    phase_space_with_pareto[outputs[0]], phase_space_with_pareto[outputs[1]], marker = 'o',
    c='pink', edgecolor='k', linewidths=0.0, s=300, alpha = 1.0, zorder = 0)

data_input.shape[0]
ax.set_title('Optimization', fontsize=24, loc='left', pad = 15)

plt.text(0.02, 0.00, f'Batches of {data_input[(data_input['Iteration'] == 1)].shape[0]}', transform=ax.transAxes, fontfamily='Calibri', fontsize = 28, ha='left', va='bottom')
plt.text(0.98, 0.00, f'Tests: {data_input.shape[0]}', transform=ax.transAxes, fontfamily='Calibri', fontsize = 28, ha='right', va='bottom')

def add_color_bar(
        axis: plt.Axes,
        segments_num: int,
        palette: str,
        label_fontsize: int
        ) -> None:
    cax = fig.add_axes([axis.get_position().x0 + 0.0, axis.get_position().y1 - 0.1 * axis.get_position().height, 1 * axis.get_position().width, 0.10 * axis.get_position().height])
    cmap = plt.get_cmap(palette)
    new_cmap = colors.ListedColormap(np.vstack(([0, 0, 0, 1], cmap(np.linspace(0, 1, cmap.N)))))
    bounds = np.linspace(0, 1, segments_num + 1)
    colornorm = colors.BoundaryNorm(bounds, new_cmap.N)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm = colornorm, cmap = new_cmap),
                      cax = cax, orientation = 'horizontal')
    tickcenters = np.array([round(i - bounds[1] / 2, 4) for i in bounds[1::]])
    ticklabels = [str(i + 1) for i in range(segments_num)]
    cb.ax.tick_params(labelsize = label_fontsize)
    cb.set_ticks(tickcenters)
    cb.set_ticklabels(ticklabels)
    # cb.set_label('Iteration', fontsize=16)
    return None

add_color_bar(
    axis = ax,
    segments_num = data_input['Iteration'].max(),
    palette = 'viridis',
    label_fontsize = 20)

# plt.tight_layout()

fig.set_edgecolor('black')
border_thickness = 6
fig.set_linewidth(border_thickness)

save_graphs = False
if save_graphs:
    dpi = 600
    export_file_name = f'test_export.png'
    plt.savefig(export_file_name, format='png', dpi=dpi, bbox_inches='tight')

plt.show()


# %% Hypervolume

fig, ax = plt.subplots(figsize=(6, 6), frameon=True)

ax.set_title('Hypervolume Improvement', fontsize=24, loc='left', pad = 15)
ax.set_xlabel('Iteration',fontsize=24, labelpad = 12)
ax.set_ylabel('Hypervolume', fontsize=24, labelpad = 12)
ax.set_xticks(range(1,10+1,1))
ax.tick_params(axis='both', which='major', labelsize=24)
# ax.set_ylim(100, 300)
# ax.set_xlim(plot_x_min, plot_x_max)

running_hypervolume = data_input['hypervolume'].iloc[(batch_size - 1)::batch_size].tolist()
ax.plot(range(1,data_input['Iteration'].max()+1,1), running_hypervolume, 'o-', color = 'k', zorder = 1, linewidth = 3)
ax.scatter(range(1,data_input['Iteration'].max()+1,1), running_hypervolume, color='cyan', linewidths = 2.0, edgecolor = 'k', s = 200, zorder = 2)

ax.plot([1, data_input['Iteration'].max()], [phase_space_hypervolume_min, phase_space_hypervolume_min], '--', color = 'k', linewidth = 3)
ax.plot([1, data_input['Iteration'].max()], [phase_space_hypervolume, phase_space_hypervolume], '--', color = 'k', zorder = 4, linewidth = 3)

fig.set_edgecolor('black')
border_thickness = 6
fig.set_linewidth(border_thickness)


save_graphs = False
if save_graphs:
    dpi = 600
    # export_file_name = f'simulation_cbbo.png'
    export_file_name = f'simulation_brs.png'
    plt.savefig(export_file_name, format='png', dpi=dpi, bbox_inches='tight')

plt.show()

# %%

# import matplotlib.pyplot as plt

def plot_histogram_types(df, column, bin_width=None, num_bins=None, bar_color='blue'):
    plt.figure(figsize=(8, 6))
    min_val = df[column].min()
    max_val = df[column].max()
    bins = np.arange(min_val - bin_width / 2, max_val + bin_width, bin_width)
    plt.hist(df[column], bins=bins, edgecolor='black', color=bar_color)
    x_ticks = np.arange(min_val, max_val + bin_width, bin_width)
    plt.xticks(x_ticks[::], rotation=90, fontsize = 8)
    plt.xlabel(column, fontsize = 22)
    plt.ylabel('Frequency', fontsize = 22)
    plt.title('Populations of subsystems')
    plt.grid(True)
    plt.show()

histdata = phase_space
# histdata = phase_space_with_pareto[(phase_space_with_pareto['Pareto'] == 'y')]
# histdata = data_input
histcolumn = 'System_Type'
plot_histogram_types(histdata, histcolumn, bin_width = 1, bar_color = 'lightblue')
print(histdata[histcolumn].value_counts().sort_index(),'\n')