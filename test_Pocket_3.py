# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 20:24:13 2022

@author: jacksonj
"""

from Pocket_3 import Pocket_3
import numpy as np


def test_Pocket_3():

    # Load the test sets
    with np.load("Pocket_3_test_sets.npz") as npz_file:

        # Obtain list of arrays
        file_names = npz_file.files
        file_name_iter = iter(file_names)

        # Check proposed solution against reference solution
        # for each test set
        for _ in range(5):
            Xs = npz_file[next(file_name_iter)]
            ys = npz_file[next(file_name_iter)]
            T = npz_file[next(file_name_iter)]
            results = Pocket_3(Xs, ys, T)
            results_ref = npz_file[next(file_name_iter)]
            print(results_ref)
            assert np.allclose(results, results_ref)

if __name__ == "__main__":
    test_Pocket_3()
    print("Success")
