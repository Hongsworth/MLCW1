import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

def read_dataset(filepath):
    """ Read in the dataset from the specified filepath

    Args:
        filepath (str): The filepath to the dataset file

    Returns:
        tuple: returns a tuple of (x, y, classes), each being a numpy array.
               - x is a numpy array with shape (N, K),
                   where N is the number of instances
                   K is the number of features/attributes
               - y is a numpy array with shape (N, ), and should be integers
                   from 0 to C-1 where C is the number of classes
               - classes : a numpy array with shape (C, ), which contains the
                   unique class labels corresponding to the integers in y
    """

    x = []
    for line in open(filepath):
        if line.strip() != "":  # handle empty rows in file
            row = line.strip().split()
            x.append(list(map(float, row)))

    x = np.array(x)
    return x


x = read_dataset("wifi_db/clean_dataset.txt")


for thing in x:
    print(thing)
# def decision_tree_learning():
