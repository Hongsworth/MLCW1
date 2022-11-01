
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng

def entropy(x, y):
    for i in range(7):
        for j in range(4):
            


LABEL_COL = 7


class Tree:
    def __init__(self):
        self.left = None
        self.right = None
        self.attribute = None
        self.value = None


def read_dataset(filepath):
    x = []
    for line in open(filepath):
        if line.strip() != "":  # handle empty rows in file
            row = line.strip().split()
            x.append(list(map(float, row)))

    x = np.array(x)
    return x


def entropy(array):
    if (len(array) == 0):
        return 0
    ones = 0
    twos = 0
    threes = 0
    fours = 0
    for element in array:
        match element[LABEL_COL]:
            case 1:
                ones += 1
            case 2:
                twos += 1
            case 3:
                threes += 1
            case 4:
                fours += 1
    ans = 0
    if (ones != 0):
        ans += ones * np.log2(ones)
    if (twos != 0):
        ans += twos * np.log2(twos)
    if (threes != 0):
        ans += threes * np.log2(threes)
    if (fours != 0):
        ans += fours * np.log2(fours)
    ans /= len(array)

    return ans


def information_gain(array, split):
    main = entropy(array)
    split1 = entropy(array[:split])
    split2 = entropy(array[split:])
    return main - (split1 * split + split2 * (len(array) - split)) / len(array)


def sort_column(array, column):
    array[array[:, column].argsort()]


def find_split(array):
    if (len(array) == 2):
        return array[0], array[1], 0, 0
    highest_gain = 0
    l_dataset = []
    r_dataset = []
    for column in range(len(array[0])-1):
        sort_column(array, column)
        for element in range(len(array)):
            if (element == 0 or array[element][column] != array[element - 1]
                    [column]):
                curr_gain = information_gain(array, element)
                if (curr_gain > highest_gain):
                    highest_gain = curr_gain
                    l_dataset = array[:element]
                    r_dataset = array[element:]
                    split = (array[element][column] + array[element - 1]
                             [column])/2
                    b_column = column
    return l_dataset, r_dataset, split, b_column


def same_labels(training_dataset):
    for x in range(len(training_dataset) - 1):
        if (training_dataset[x][LABEL_COL] !=
            training_dataset[x+1][LABEL_COL]):
            return False
    return True


def decision_tree_learning(training_dataset, depth):
    curr = Tree()
    if (len(training_dataset.shape) == 1):
        curr.value = training_dataset[LABEL_COL]
        return (curr, depth)
    if (same_labels(training_dataset)):
        curr.value = training_dataset[0][LABEL_COL]
        return (curr, depth)
    l_dataset, r_dataset, split, b_column = find_split(training_dataset)
    curr.left, l_depth = decision_tree_learning(l_dataset, depth + 1)
    curr.right, r_depth = decision_tree_learning(r_dataset, depth + 1)
    curr.value = "x" + str(b_column) + " < " + str(split)
    return (curr, max(l_depth, r_depth))


dataset = read_dataset("wifi_db/clean_dataset.txt")

root = decision_tree_learning(dataset, 0)
print("stop")

tree.plot_tree(clf)
