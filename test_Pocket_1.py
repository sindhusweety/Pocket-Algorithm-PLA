# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 13:03:10 2023

@author: jacksonj
"""

from Pocket_1 import Pocket_1
import numpy as np

def test_Pocket_1():
    # Load the test sets
    with np.load("Pocket_1_test_sets.npz") as npz_file:

        # Obtain list of arrays
        file_names = npz_file.files
        file_name_iter = iter(file_names)

        # Check proposed solution against reference solution
        # for each test set
        for _ in range(5):
            X = npz_file[next(file_name_iter)]
            y = npz_file[next(file_name_iter)]
            T = npz_file[next(file_name_iter)]
            w = Pocket_1(X, y, T)
            w_ref = npz_file[next(file_name_iter)]

            print(w, w_ref)
            assert np.allclose(w, w_ref)

    
if __name__ == "__main__":
    test_Pocket_1()
    print("Success")