# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 12:16:02 2025

@author: Timet
"""

def change_directory_to_current_py_file():
    try:
        import os
        os.chdir(os.path.split(os.path.abspath(__file__))[0])
    except Exception as e:
        print(f'\nDirectory change or os module error:\n\n{e}')

change_directory_to_current_py_file()

import numpy as np

from package_BBO_functions import import_excel
from package_BBO_functions import extract_elements
from package_BBO_functions import calculate_medoids

file_name_space = 'compositionSpace_htmdec_1000_withtype'
init_plusplus = False


phase_space = import_excel(f'{file_name_space}.xlsx')
elements = extract_elements(phase_space)
data_input = calculate_medoids(data = phase_space,
                               dimensions = elements,
                               clusters = 50,
                               random_state = np.random.randint(0, 1e7),
                               # random_state = 100,
                               init_plusplus = init_plusplus)


# data_input.to_excel('compositionSpace_htmdec_10000_withtype.xlsx')
