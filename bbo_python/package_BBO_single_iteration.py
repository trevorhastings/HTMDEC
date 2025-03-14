# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 16:04:28 2025

@author: Timet
"""

def change_directory_to_current_py_file():
    try:
        import os
        os.chdir(os.path.split(os.path.abspath(__file__))[0])
    except Exception as e:
        print(f'\nDirectory change or os module error:\n\n{e}')
        
change_directory_to_current_py_file()

import pandas as pd
import numpy as np
# import time # e.g. 
'''
t0 = time.time()
t1 = time.time()
print(f'\n{round(t1-t0, 2)} seconds elapsed for ____\n')
'''

from package_BBO_functions import setup_single_folder, import_excel
from package_BBO_functions import remove_closest_points
from package_BBO_functions import create_input_space_with_only_elements
from package_BBO_functions import hypervolume, pareto_maximum
from package_BBO_functions import extract_elements, latin_hypercube_sampling
from package_BBO_functions import mean_covariance, acquisition, calculate_medoids

file_name_space = 'compositionSpace_htmdec_1000_withtype'
phase_space = import_excel(f'{file_name_space}.xlsx')
outputs = ['PROP RT Density (g/cm3)', 'PROP RT Heat Cap (J/(mol K))']

phase_space_size = phase_space.shape[0]
batch_start_size = 20
iteration_current = 2

file_name_input_pattern = f'inputSpace_from{phase_space_size}_size'

if iteration_current == 1:
    opt_folder = setup_single_folder(
        base_folder='BBO Python Package', 
        category='Test Optimization Iteration',
        sim_type='testBOloop_specificmetadata',
    )
    data_input = create_input_space_with_only_elements(
        phase_space,
        outputs,
        batch_start_size)
    data_input.to_excel(f'{opt_folder}/{file_name_input_pattern}{batch_start_size}_iter1.xlsx')

print(f'Starting hypervolume: {data_input["hypervolume"].max()}')
print(f'Outputs max:\n{data_input[outputs].max()}')
print(f'Outputs min:\n{data_input[outputs].min()}')

file_name_input_pattern = f'inputSpace_from{phase_space_size}_size{batch_start_size}_iter'

folder_name_input = opt_folder

file_name_input = f'{folder_name_input}/{file_name_input_pattern}{iteration_current}'

hyperparameter_file_name = 'hyperparameter'
gaussian_process_number = 100
lengthscale_min, lengthscale_max = 0.05, 0.95
batch_size = 20

dimensions_output_number = len(outputs)

elements = extract_elements(phase_space)
dimensions_clusters = elements
dimensions_input_names = elements
init_plusplus = False

dimensions_input_number = data_input[dimensions_input_names].to_numpy().shape[1]

if iteration_current == 1:
    hyperparameter_dimensions_input_number = dimensions_input_number
    kernel_lengthscales = latin_hypercube_sampling(number_of_samples = gaussian_process_number,
                              dimensions = hyperparameter_dimensions_input_number,
                               ls_min = lengthscale_min,
                               ls_max = lengthscale_max,
                                # seed = 0
                               )
    np.savetxt(f'{folder_name_input}/{hyperparameter_file_name}.csv', kernel_lengthscales, delimiter=',')

if iteration_current != 1:
    file_name_input = f'{folder_name_input}/{file_name_input_pattern}{iteration_current}'
    data_input = import_excel(f'{file_name_input}.xlsx')
    kernel_lengthscales = np.genfromtxt(f'{folder_name_input}/{hyperparameter_file_name}.csv', delimiter=',')

'''Remainder of testable points with exact values'''
phase_space_untested = phase_space[
~phase_space.set_index(dimensions_input_names)
.index.isin(data_input.set_index(dimensions_input_names).index)]
'''Remainder of testable points with nearest values'''
# phase_space_untested = remove_closest_points(data_input, phase_space_with_pareto, dimensions_input_names)

dimensions_input = data_input[dimensions_input_names].to_numpy()
dimensions_input_untested = phase_space_untested[dimensions_input_names].to_numpy()
dimensions_output = data_input[outputs].to_numpy()

'''No normalization'''
# dimensions_output_normalized = dimensions_output
'''Approximate order of magnitude'''
# dimensions_output_normalized = dimensions_output / np.array([1, 10])
'''min max normalization from 1 to 2 based on the first iteration"s data '''
dimensions_output_normalized = 1 + (dimensions_output - np.min(dimensions_output[:batch_size,:], axis=0)) / (np.max(dimensions_output[:batch_size,:], axis=0) - np.min(dimensions_output[:batch_size,:], axis=0))

dimensions_output_error = np.zeros((data_input.shape[0], dimensions_output_number)) + 0.001

pareto_set_outputs_normalized = pareto_maximum(dimensions_output_normalized)

kernel_variances = (np.max(dimensions_output_normalized, axis = 0) * (1/2) * 1.22) ** 2

optimization = 'bayesian'
# optimization = 'random'
if optimization == 'bayesian':
    
    candidates = np.zeros([gaussian_process_number, dimensions_input_number])
    candidates_backup = np.zeros([gaussian_process_number, dimensions_input_number])
    improvements = np.zeros(gaussian_process_number)
    for gaussian_process_index in range(gaussian_process_number):
        gaussian_processes: list[dict] = []
        for dimension in range(dimensions_output_number):
            gaussian_process = {'input_training_data': dimensions_input,
                                'output_training_data': dimensions_output[:, dimension],
                                'output_data_error': dimensions_output_error[:, dimension],
                                'kernel_variance': kernel_variances[dimension],
                                'kernel_lengthscales': kernel_lengthscales[gaussian_process_index,:]}
            gaussian_processes.append(gaussian_process)
        
        means = []
        sigmas = []
        for i, gaussian_process in enumerate(gaussian_processes):
            mean, sigma = mean_covariance(gaussian_process, dimensions_input_untested)
            means.append(mean)
            sigmas.append(sigma)
        means_ehvi = np.column_stack(means)
        sigmas_ehvi = np.column_stack((np.abs(np.diag(sigmas[0])) ** 0.5, np.abs(np.diag(sigmas[1])) ** 0.5))
        # goal = np.ones(dimensions_output_number)
        ehvi = acquisition(means_ehvi, sigmas_ehvi, pareto_set_outputs_normalized)
        
        candidates[gaussian_process_index] = dimensions_input_untested[np.argmax(ehvi)]
        improvements[gaussian_process_index] = np.max(ehvi)
        ehvi_backup = np.copy(ehvi)
        ehvi_backup[np.argmax(ehvi)] = -np.inf
        candidates_backup[gaussian_process_index] = dimensions_input_untested[np.argmax(ehvi_backup)]
    
    candidates_unique = pd.merge(
        pd.DataFrame(np.unique(candidates, axis=0), columns = dimensions_input_names),
        phase_space_untested, on = dimensions_input_names, how = 'left')
    
    if len(candidates_unique) >= batch_size:
        candidates_batch = calculate_medoids(data = candidates_unique,
                                              dimensions = dimensions_clusters,
                                              clusters = batch_size,
                                              random_state = 0,
                                              init_plusplus = init_plusplus)
    else:
        print(f'Not enough candidates: {candidates_unique.shape[0]}')
        candidates_backup_unique = pd.merge(
            pd.DataFrame(np.unique(candidates_backup, axis = 0), columns = dimensions_input_names),
            phase_space_untested, on = dimensions_input_names, how = 'left')
        print(f'Added additional candidates: {candidates_backup_unique.shape[0]} \
              \nGetting extra medoids')
        if len(candidates_backup_unique) >= (batch_size - len(candidates_unique)):
            candidates_extra = calculate_medoids(data = candidates_backup_unique,
                                                  dimensions = dimensions_clusters,
                                                  clusters = batch_size - len(candidates_unique),
                                                  random_state = 0,
                                                  init_plusplus = init_plusplus)
            candidates_batch = pd.concat([candidates_unique, candidates_extra],
                                          ignore_index = True)
        
        print(f'Total candidates: {candidates_batch.shape[0]}')

if optimization == 'random':
    candidates_batch = phase_space_untested.sample(n = batch_size, replace = False)

iteration_current = data_input['Iteration'].max()
iteration_next = iteration_current + 1

candidates_batch['Iteration'] = iteration_next
candidates_batch['Alloy'] = [i + 1 + batch_size * iteration_current for i in range(batch_size)]

candidates_pareto_set = pareto_maximum(candidates_batch[outputs].to_numpy())
candidates_hypervolume = hypervolume(candidates_pareto_set)

data_input_next = pd.concat([data_input, candidates_batch], ignore_index = True)
data_input_next_pareto_set = pareto_maximum(data_input_next[outputs].to_numpy())
data_input_next_hypervolume = hypervolume(data_input_next_pareto_set)
max_index_current = batch_size * iteration_current
max_index_next = batch_size * iteration_next
data_input_next.loc[max_index_current:max_index_next, 'hypervolume current'] = candidates_hypervolume
data_input_next.loc[max_index_current:max_index_next, 'hypervolume'] = data_input_next_hypervolume
print(f'Iteration {iteration_next} predicted and exported\n'
      f'Hypervolume: from {round(data_input_next["hypervolume"].iloc[:max_index_current].max(),2)} '
          f'to {round(data_input_next["hypervolume"].iloc[:max_index_next].max(),2)}')

data_input_next.to_excel(f'{folder_name_input}/{file_name_input_pattern}{iteration_next}.xlsx')
if optimization == 'bayesian':
    candidates_unique.to_excel(f'{folder_name_input}/candidates_unique_{iteration_current}_to_{iteration_next}.xlsx')

print('Optimization iteration completed')