
# import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
# from Find_split_rough import decision_tree_learning

LABEL_COL = 7


# ____________________________SPLITTING FUNCTIONS______________________________

def k_fold_split(n_splits, n_instances, random_generator=default_rng()):
    shuffled_indices = random_generator.permutation(n_instances)
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
    return find_accuracy(predicted_labels, actual_labels)


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

    # accuracy is given as a percentage
    accuracy = (correct_samples/total_samples) * 100

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


def main():
    
    dataset = read_dataset("filepath")
    # split data into 10 folds
    # for loop through the combination of folds 
    # in each loop: train tree, evaluate unpruned tree, run prune function. ensure we are aggregating the confusion matrix 
    
