# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:04:42 2025

@author: Timet
"""
import pandas as pd
import numpy as np
from scipy.stats import norm
from statistics import NormalDist
import os

def timeit(func):
    from functools import wraps
    @wraps(func)
    def timed(*args, **kwargs):
        import time
        t0 = time.monotonic()
        result = func(*args, **kwargs)
        t1 = time.monotonic()
        print(f'Time elapsed for {func.__name__}: {(t1 - t0):.3f} seconds')
        return result
    return timed

def import_excel(
        file_name: str
        ) -> pd.DataFrame():
    
    with pd.ExcelFile(file_name) as xlsx:
        sheet_names = xlsx.sheet_names
        data = pd.read_excel(xlsx, sheet_name=sheet_names[0])
    data = data.filter(regex='^(?!Unnamed|Index)')
    
    return data

def pareto_maximum(
        data_points: np.ndarray
        ) -> np.ndarray:
    
    data_points = data_points[data_points.sum(1).argsort()[::-1]]
    is_not_dominated = np.ones(data_points.shape[0], dtype = bool)
    for i in range(data_points.shape[0]):
        n = data_points.shape[0]
        if i >= n:
            break
        is_not_dominated[i+1:n] = (data_points[i+1:] > data_points[i]).any(1)
        data_points = data_points[is_not_dominated[:n]]
        is_not_dominated = np.array([True] * len(data_points))
        
    return data_points

def extract_elements(
        data: pd.DataFrame
        ) -> list[str]:
    
    elements = [col for col in data.columns if len(col) <= 2]
    print(f'\nExample rows:\n{data.head().iloc[:,:len(elements)]}')
    
    return elements

def random_sampling(number_of_samples: int, dimensions: int, ls_min: float, ls_max: float) -> list:
    random_numbers = np.random.rand(number_of_samples, dimensions)
    random_numbers = random_numbers * ls_max + ls_min
    return random_numbers


def latin_hypercube_sampling_scipy(
        number_of_samples: int,
        dimensions: int,
        ls_min: float,
        ls_max: float,
        seed: int = None
        ) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    
    from scipy.stats import qmc
    sampler = qmc.LatinHypercube(d = dimensions)
    random_numbers = sampler.random(n = number_of_samples)
    random_numbers = random_numbers * (ls_max - ls_min) + ls_min
    return random_numbers

def latin_hypercube_sampling(
        number_of_samples: int,
        dimensions: int,
        ls_min: float,
        ls_max: float,
        seed: int = None
        ) -> np.ndarray:
    
    if seed is not None:
        np.random.seed(seed)        
    random_numbers = np.zeros((number_of_samples, dimensions))
    segment = (ls_max - ls_min) / number_of_samples
    
    for i in range(dimensions):
        perm = np.random.permutation(number_of_samples)
        for j in range(number_of_samples):
            random_numbers[j, i] = ls_min + segment * (perm[j] + np.random.rand())
    
    return random_numbers

def hypervolume(
        pareto_set: np.ndarray
        ) -> float:
    pareto_set = pareto_set[pareto_set[:, 0].argsort()]
    
    def calculate_base_HV(
            pareto_subset: np.ndarray
            ) -> float:
        if pareto_subset.shape[0] == 1:
            return np.prod(pareto_subset[0])
        
        volume_at_index = np.prod(pareto_subset[0])
        
        coordinates_offset = np.zeros((pareto_subset.shape[0] - 1, pareto_subset.shape[1]))
        for index in range(1, pareto_subset.shape[0]):
            coordinates_offset[index - 1] = np.min([pareto_subset[0], pareto_subset[index]], axis = 0)
        coordinates_offset = pareto_maximum(coordinates_offset)
        volume_excess = calculate_base_HV(coordinates_offset)
        
        volume_base = calculate_base_HV(pareto_subset[1:])
        
        return volume_at_index - volume_excess + volume_base
    
    hv = calculate_base_HV(pareto_set)
    
    return hv

def hypervolume_smallest(
        dataset = np.ndarray,
        num_datapoints = int,
        ) -> float:
    final_array = np.zeros((num_datapoints, dataset.shape[1]))
    filled_rows = 0
    while filled_rows < num_datapoints:
        result = pareto_minimum(dataset)
        num_rows = result.shape[0]
        rows_to_fill = min(num_rows, num_datapoints - filled_rows)
        final_array[filled_rows:filled_rows + rows_to_fill, :] = result[:rows_to_fill, :]
        filled_rows += rows_to_fill        
        mask = np.all(~np.isin(dataset, result).all(axis=1).reshape(-1, 1), axis=1)
        dataset = dataset[mask]        
        if filled_rows >= 20:
            break
    return hypervolume(final_array)

def cholesky_decomposition(
        square_matrix: np.ndarray
        ) -> np.ndarray:
    try:
        L = np.linalg.cholesky(square_matrix)
        return L
    except np.linalg.LinAlgError:
        raise Exception("Cholesky factorization failed. Matrix may not be positive definite.")
        
def kernel(
        dataset_1: np.ndarray,
        dataset_2: np.ndarray,
        kernel_variance: float,
        kernel_lengthscales: list[float]
        ) -> np.ndarray:
    '''Squared Exponential aka Gaussian aka Radial Basis'''
    
    kernel_matrix = np.zeros((dataset_1.shape[0], dataset_2.shape[0]))
    scaled_lengthscales = np.broadcast_to(kernel_lengthscales, (dataset_1.shape[0], kernel_lengthscales.size))
    
    for index in range(dataset_2.shape[0]):
        diff = (dataset_1 / scaled_lengthscales - dataset_2[index, :] / kernel_lengthscales) ** 2
        kernel_matrix[:, index] = kernel_variance * np.exp(-0.5 * np.sum(diff, axis=1))
        
    return kernel_matrix

def mean_covariance(
        gaussian_process: dict[str, np.ndarray],
        dimensions_input_untested: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
    
    Kxz = kernel(dataset_1 = gaussian_process['input_training_data'],
                 dataset_2 = dimensions_input_untested,
                 kernel_variance = gaussian_process['kernel_variance'],
                 kernel_lengthscales = gaussian_process['kernel_lengthscales'])
    Kzz = kernel(dataset_1 = dimensions_input_untested,
                 dataset_2 = dimensions_input_untested,
                 kernel_variance = gaussian_process['kernel_variance'],
                 kernel_lengthscales = gaussian_process['kernel_lengthscales'])
    
    # Optional test for data correlation under the assumption of equal training uncertainty
    # Kxx = np.full((len(gaussian_process['input_training_data']), 1), gaussian_process['kernel_variance'])
    Kxx = kernel(dataset_1=gaussian_process['input_training_data'],    
                  dataset_2=gaussian_process['input_training_data'],
                  kernel_variance=gaussian_process['kernel_variance'],
                  kernel_lengthscales=gaussian_process['kernel_lengthscales'])

    R = cholesky_decomposition(Kxx + np.diag(gaussian_process['output_data_error'] ** 2))
    ad = np.linalg.solve(R, gaussian_process['output_training_data'])
    bd = np.linalg.solve(R, Kxz)
    
    means = bd.T.dot(ad)
    cov = Kzz - bd.T.dot(bd)
    
    return means, cov

def acquisition_old(means, sigmas, goal, dimensions_output_number, pareto):
    ref = np.zeros(dimensions_output_number)
    # Determine the dimensions
    n_points, n_obj = means.shape

    # Adjust means and pareto according to the goal
    for i in range(goal.size):
        if goal[i] == 1:
            means[:, i] = -1 * means[:, i]
            pareto[:, i] = -1 * pareto[:, i]

    # Sort the Pareto front by the first objective
    pareto = pareto[np.argsort(pareto[:, 0]), :]

    # Initialize EHVI array
    ehvi = np.zeros(n_points)

    # Compute EHVI for each test point
    for i in range(n_points):
        box = 1
        ref = np.zeros(n_obj)
        for j in range(n_obj):
            s = (ref[j] - means[i, j]) / sigmas[i, j]
            box *= (ref[j] - means[i, j]) * norm.cdf(s) + sigmas[i, j] * norm.pdf(s)
        hvi = recursive_iteration_old(means[i, :], sigmas[i, :], ref, pareto)
        ehvi[i] = box - hvi
    
    return ehvi

def recursive_iteration_old(means, sigmas, ref, pareto):
    improvement = 0
    hvi_temp = 1
    while pareto.shape[0] > 1:
        s_up = (ref - means) / sigmas # again, don't need ref point and negative
        s_low = (pareto[0, :] - means) / sigmas
        up = (ref - means) * norm.cdf(s_up) + sigmas * norm.pdf(s_up)
        low = (pareto[0, :] - means) * norm.cdf(s_low) + sigmas * norm.pdf(s_low)
        hvi_temp *= np.prod(up - low, axis=0)
        pareto = np.maximum(pareto[0, :], pareto[1:, :])
        pareto = pareto_minimum(pareto)
    improvement = hvi_temp
    
    return improvement

def acquisition(
        means: np.ndarray,
        sigmas: np.ndarray,
        pareto: np.ndarray
        ) -> np.ndarray:
    '''Expected Hypervolume Improvement (EHVI)'''
    
    pareto = pareto[np.argsort(pareto[:, 0]), :]
    ehvi = np.zeros(means.shape[0])
    # hviall = np.zeros(means.shape[0])
    
    for i in range(ehvi.shape[0]):
        box = 1
        for j in range(means.shape[1]):
            s = means[i, j] / sigmas[i, j]
            box *= means[i, j] * NormalDist().cdf(s) + sigmas[i, j] * NormalDist().pdf(s)
        # hvi = recursive_iteration2(means[i, :], sigmas[i, :], pareto)
        if box >= max(ehvi[:i], default = 0):
            hvi = recursive_iteration2(means[i, :], sigmas[i, :], pareto)
        else:
            hvi = box
        ehvi[i] = box - hvi
        
    return ehvi

def recursive_iteration2(
        means: np.ndarray,
        sigmas: np.ndarray,
        pareto: np.ndarray
        ) -> float:
    
    hvi_temp = 1
    while pareto.shape[0] > 1:
        s_up = means / sigmas
        s_low = (means - pareto[-1, :]) / sigmas
        up = means * np.array([NormalDist().cdf(value) for value in s_up]) + sigmas * np.array([NormalDist().pdf(value) for value in s_up])
        low = (means - pareto[-1, :]) * np.array([NormalDist().cdf(value) for value in s_low]) + sigmas * np.array([NormalDist().pdf(value) for value in s_low])
        hvi_temp *= np.prod(up - low, axis = 0)
        pareto = np.minimum(pareto[-1, :], pareto[:-1, :])
        pareto = pareto_maximum(pareto)
    improvement = hvi_temp
    
    return improvement

def pareto_minimum(
        data_points: np.ndarray
        ) -> np.ndarray:
    
    data_points = data_points[data_points.sum(1).argsort()]
    is_not_dominated = np.ones(data_points.shape[0], dtype = bool)
    for i in range(data_points.shape[0]):
        n = data_points.shape[0]
        if i >= n:
            break
        is_not_dominated[i+1:n] = (data_points[i+1:] < data_points[i]).any(1)
        data_points = data_points[is_not_dominated[:n]]
        is_not_dominated = np.array([True] * len(data_points))
        
    return data_points

def calculate_medoids(
        data: pd.DataFrame,
        dimensions: list[str],
        clusters: int,
        random_state: int,
        init_plusplus: bool
        ) -> pd.DataFrame:

    from sklearn_extra.cluster import KMedoids
    
    if init_plusplus == False:
        medoid_fit = KMedoids(n_clusters = clusters,
                              init = 'random',
                              random_state = random_state)
    if init_plusplus == True:
        medoid_fit = KMedoids(n_clusters = clusters,
                              init ='k-medoids++',
                              random_state = random_state)
    medoid_fit = medoid_fit.fit(data[dimensions])
    medoid_inputs = pd.DataFrame(medoid_fit.cluster_centers_)
    medoid_inputs.columns = dimensions
    medoids = pd.merge(medoid_inputs, data, on=dimensions,how='inner')
    
    return medoids

def make_directory(folder_name: str) -> None:
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return None

def setup_folders(
        base_folder: str,
        category: str,
        sim_type: str,
        optimization_number_start: int,
        optimization_number_end: int,
        # num_optimizations: int
        ) -> None:
    folder_tree = {
        0: base_folder,
        1: category,
        2: sim_type
    }
        
    simulation_folder = f'{folder_tree[1]}/{folder_tree[2]}'
    make_directory(simulation_folder)
    opt_folders = []
    
    for opt_num in range(optimization_number_start, optimization_number_end + 1):
        folder_tree[3] = f'set_{opt_num}'
        opt_folder = f'{simulation_folder}/{folder_tree[3]}'
        make_directory(opt_folder)
        opt_folders.append(opt_folder)
        print(f"Created folder: {opt_folder}")
    return opt_folders

def setup_single_folder(
        base_folder: str,
        category: str,
        sim_type: str
        ) -> None:

    folder_tree = {
        0: base_folder,
        1: category,
        2: sim_type
    }

    simulation_folder = f'{folder_tree[1]}/{folder_tree[2]}'
    make_directory(simulation_folder)
    
    print(f"Created folder: {simulation_folder}")
    return simulation_folder

def find_and_add_maximal_pareto_set(
        dataset_all: pd.DataFrame,
        objective_columns: list[str]
        ) -> pd.DataFrame:
    
    pareto_set = pareto_maximum(dataset_all[objective_columns].to_numpy())
    pareto_set_dataset_rows = pd.merge(pd.DataFrame(pareto_set, columns = objective_columns),
                                       dataset_all,
                                       on = objective_columns, how = 'left')
    pareto_set_dataset_rows['Pareto'] = 'y'
    dataset_all_with_pareto_set = pd.merge(dataset_all, pareto_set_dataset_rows, how = 'outer')
    dataset_all_with_pareto_set = dataset_all_with_pareto_set.fillna('n')
    
    for i, objective in enumerate(objective_columns):
        dataset_all_with_pareto_set[f'objective_{i + 1}'] = objective
        
    return dataset_all_with_pareto_set

def create_input_space_with_only_elements(
        phase_space: pd.DataFrame,
        outputs: list[str],
        batch_start_size: int
        ) -> pd.DataFrame:
    
    elements = extract_elements(phase_space)
    dimensions_clusters = elements
    init_plusplus = False
    
    data_input = calculate_medoids(data = phase_space,
                                   dimensions = dimensions_clusters,
                                   clusters = batch_start_size,
                                    # random_state = np.random.randint(0, 1e7),
                                    random_state = 555555,
                                   init_plusplus = init_plusplus)

    data_input['hypervolume'] = hypervolume(pareto_maximum(data_input[outputs].to_numpy()))
    data_input['hypervolume current'] = hypervolume(pareto_maximum(data_input[outputs].to_numpy()))
    data_input['Iteration'] = 1
    data_input['Alloy'] = [alloy + 1 for alloy in range(batch_start_size)]
    
    return data_input

def create_input_space(
        phase_space: pd.DataFrame,
        outputs: list[str],
        batch_start_size: int,
        dimensions_arb_nonzero: bool
        ) -> pd.DataFrame:
    
    elements = extract_elements(phase_space)
    dimensions_clusters = elements
    init_plusplus = False
    if dimensions_arb_nonzero == True:
        dimensions_clusters = elements + ['System_Type']
        init_plusplus = True
    
    data_input = calculate_medoids(data = phase_space,
                                   dimensions = dimensions_clusters,
                                   clusters = batch_start_size,
                                    # random_state = np.random.randint(0, 1e7),
                                    random_state = 555555,
                                   init_plusplus = init_plusplus)

    data_input['hypervolume'] = hypervolume(pareto_maximum(data_input[outputs].to_numpy()))
    data_input['hypervolume current'] = hypervolume(pareto_maximum(data_input[outputs].to_numpy()))
    data_input['Iteration'] = 1
    data_input['Alloy'] = [alloy + 1 for alloy in range(batch_start_size)]
    
    return data_input

def remove_closest_points(data_input_perturbed, phase_space_with_pareto, dimensions_input_names):
    from scipy.spatial import cKDTree
    
    perturbed_points = data_input_perturbed[dimensions_input_names].values
    phase_space_points = phase_space_with_pareto[dimensions_input_names].values

    tree = cKDTree(phase_space_points)

    _, closest_indices = tree.query(perturbed_points, k=1)

    closest_indices = np.unique(closest_indices)

    phase_space_untested = phase_space_with_pareto.drop(phase_space_with_pareto.index[closest_indices])

    return phase_space_untested