# import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng

seed = 60012
rg = default_rng(seed)

#_____________________________BUILDING TREE_______________________________________

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
        if element[LABEL_COL] == 1:
            ones += 1
        elif element[LABEL_COL] == 2:
            twos += 1
        elif element[LABEL_COL] == 3:
            threes += 1
        elif element[LABEL_COL] == 4:
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
    split = 0 
    b_column = 0
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
    #if (len(training_dataset.shape) == 1):
    if (np.shape(training_dataset)[0] == 1):
        curr.value = training_dataset[LABEL_COL]
        return (curr, depth)
    if (same_labels(training_dataset)):
        curr.value = training_dataset[0][LABEL_COL]
        return (curr, depth)
    l_dataset, r_dataset, curr.value, curr.attribute = find_split(training_dataset)
    curr.left, l_depth = decision_tree_learning(l_dataset, depth + 1)
    curr.right, r_depth = decision_tree_learning(r_dataset, depth + 1)
    return (curr, max(l_depth, r_depth))

#_____________________________PRUNING FUNCTIONS________________________________

def prune_tree(test_db, tree):
    
    if (tree.left != None):
        prune_tree (test_db, tree.left)
    
    if (tree.right != None):
        prune_tree (test_db, tree.right)

    if(tree.left.attribute == None and tree.right.attribute == None):
        l_acc = evaluate(test_db, tree.left)
        r_acc = evaluate(test_db, tree.right)
        if (l_acc < r_acc):
            tree.value = tree.right.value
            tree.attribute = None
            del tree.left, tree.right

        else:
            tree.value = tree.left.value
            tree.attribute = None
            del tree.left, tree.right

        
# ____________________________SPLITTING FUNCTIONS______________________________

def k_fold_split(n_splits, n_instances, random_generator=default_rng()):
    # generate a random permutation of indices from 0 to n_instances
    shuffled_indices = random_generator.permutation(n_instances)

    # split shuffled indices into almost equal sized splits
    split_indices = np.array_split(shuffled_indices, n_splits)

    return split_indices


def split_labels_from_dataset(dataset):
    labels = []
    for row in dataset:
        print(row)
        labels.append(row[7])

    return labels


# ____________________________EVALUATION_FUNCTION______________________________

def eval_helper(data, node):
    if node.attribute is None:
        return node.value
    if data[node.attribute] < node.value:
        return eval_helper(data, node.left)
    return eval_helper(data, node.right)


def evaluate(test_db, trained_tree):
    predicted_labels = []
    actual_labels = []
    for data in test_db:
        # passes in test data into tree and tree produces an array of predicted
        # labels
        predicted_labels.push(eval_helper(data, trained_tree))
        actual_labels.push(data[LABEL_COL])

    # pass the array of predicted labels and actual labels into the find
    # accuracy function
    accuracy = find_accuracy(predicted_labels, actual_labels)
    conf_matrix = create_confusion_matrix(predicted_labels, actual_labels)

    return accuracy, conf_matrix


# _____________________________EVALUATION_METRICS_______________________________

def create_confusion_matrix(predicted_labels, actual_labels):

    classes = np.unique(actual_labels)

    conf_matrix = np.zeros((len(classes), len(classes)))

    for i in range(len(classes)):

        for j in range(len(classes)):
            conf_matrix[i, j] = np.sum((actual_labels == classes[i]) &
                                       (predicted_labels == classes[j]))

    return conf_matrix


def find_accuracy(predicted_labels, actual_labels):

    correct_samples = 0
    total_samples = len(predicted_labels)
    accuracy = 0

    conf_matrix = create_confusion_matrix(predicted_labels, actual_labels)
    classes = np.unique(actual_labels)

    for i in range(len(classes)):
        for j in range(len(classes)):
            if i == j:
                correct_samples = correct_samples + conf_matrix[i, j]

    accuracy = correct_samples/total_samples

    return accuracy


def find_recall(class_num, predicted_labels, actual_labels):

    recall = 0
    total_class_samples = 0
    conf_matrix = create_confusion_matrix(predicted_labels, actual_labels)

    correct_class_samples = conf_matrix[class_num, class_num]

    for sample in actual_labels:
        if sample == class_num:
            total_class_samples += 1

    recall = correct_class_samples / total_class_samples

    return recall


def find_precision(class_num, predicted_labels, actual_labels):

    precision = 0
    total_class_samples = 0
    conf_matrix = create_confusion_matrix(predicted_labels, actual_labels)

    correct_class_samples = conf_matrix[class_num, class_num]

    for sample in predicted_labels:
        if sample == class_num:
            total_class_samples += 1

    precision = correct_class_samples / total_class_samples

    return precision


def find_F1(class_num, predicted_labels, actual_labels):

    f_measure = 0
    recall = find_recall(class_num, predicted_labels, actual_labels)
    precision = find_precision(class_num, predicted_labels, actual_labels)

    f_measure = (2*recall*precision)/(precision + recall)

    return f_measure


def main(filename):
    
    cumalative_conf_matrix = np.zeros((4,4))
    dataset = read_dataset(filename)
    
    # split data into 10 folds
    split_indices = k_fold_split(10, dataset, rg)

    size = np.shape(split_indices[0])[1]
  
    for k in range(10):
        arr1 = np.empty((0,size), int)
        arr2 = np.empty((0,size), int)
      
        # pick k as test
        test_indices = split_indices[k]
        # combine remaining splits as train  
        for i in range(k):
          arr1 = np.append(arr1, split_indices[i], axis=0)
          
        for i in range(k+1, 10):
          arr2 = np.append(arr2, split_indices[i], axis=0)
        
        train_indices = np.append(arr1, arr2, axis=0)
  
          # for validation:
          # validate_indicies = split_indicies[k+1] ??
          # train_indices = np.hstack(split_indices[:k+1] + split_indices[k+2:])
  
          # combine remaining splits as train    
         
  
        trained_tree = decision_tree_learning(train_indices, 0)
        accuracy, conf_matrix = evaluate(test_indices, trained_tree) 
        cumalative_conf_matrix += conf_matrix

    average_conf_matrix = cumalative_conf_matrix / 10

    return average_conf_matrix

filename = "clean_dataset.txt"
print(main(filename))

# in each loop: train tree, evaluate unpruned tree, run prune function. ensure
# we are aggregating the confusion matrix
