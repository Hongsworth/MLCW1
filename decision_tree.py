import numpy as np
from numpy.random import default_rng


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
    y_labels = []
    for line in open(filepath):
        if line.strip() != "":  # handle empty rows in file
            row = line.strip().split()
            x.append(list(map(float, row[:-1])))
            y_labels.append(row[-1])

    [classes, y] = np.unique(y_labels, return_inverse=True)

    x = np.array(x)
    y = np.array(y)
    return (x, y, classes)


(x, y, classes) = read_dataset("wifi_db/clean_dataset.txt")
print(x.shape)
print(y.shape)
print(classes)

# def decision_tree_learning():
