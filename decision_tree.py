
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng


class Tree:
    def __init__(self):
        self.left = None
        self.right = None
        self.data = None


def read_dataset(filepath):
    x = []
    for line in open(filepath):
        if line.strip() != "":  # handle empty rows in file
            row = line.strip().split()
            x.append(list(map(float, row)))

    x = np.array(x)
    return x


def entropy(array):
    ones = 0
    twos = 0
    threes = 0
    fours = 0
    for element in array:
        match element[7]:
            case 1:
                ones += 1
            case 2:
                twos += 1
            case 3:
                threes += 1
            case 4:
                fours += 1
    return -(ones * np.log2(ones) + twos * np.log2(twos) + threes *
             np.log2(threes) + fours * np.log2(fours))


def information_gain(array, split):
    main = entropy(array)
    split1 = entropy(array[:split])
    split2 = entropy(array[split:])
    return main - (split1 * split + split2 * (len(array) - split)) / len(array)


def sort_column(array, column):
    array[array[:, column].argsort()]


def find_split(array):
    highest_gain = 0
    l_dataset = []
    r_dataset = []
    for column in range(7):
        sort_column(array, column)
        for element in range(len(array)):
            if (element == 0 or array[element] != array[element - 1]):
                curr_gain = information_gain(element)
                if (curr_gain > highest_gain):
                    highest_gain = curr_gain
                    l_dataset = array[:element]
                    r_dataset = array[element:]
    return l_dataset, r_dataset


def same_labels(training_dataset):
    for x in range(len(training_dataset) - 1):
        if (training_dataset[x][7] != training_dataset[x+1][7]):
            return False
    return True


def decision_tree_learning(training_dataset, depth):
    if (same_labels(training_dataset)):
        return (leaf, depth)
    l_dataset, r_dataset = find_split(training_dataset)
    l_branch, l_depth = decision_tree_learning(l_dataset, depth + 1)
    r_branch, r_depth = decision_tree_learning(r_dataset, depth + 1)
    return (node, max(l_depth, r_depth))


x = read_dataset("wifi_db/clean_dataset.txt")
