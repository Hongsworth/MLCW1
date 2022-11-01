
# import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
import Find_split_rough

# Fods
#_____________________________SPLITTING FUNCTIONS______________________________

def k_fold_split(n_splits, n_instances, random_generator=default_rng()):
     
     shuffled_indices = random_generator.permutation(n_instances)
     
     split_indices = np.array_split(shuffled_indices, n_splits)
     
     return split_indices
# from numpy.random import default_rng

def split_labels_from_dataset(dataset):
    labels = []
    for row in dataset:
        print(row)
        labels.append(row[7])
    
    return labels

#_____________________________EVALUATION_FUNCTION______________________________
def predict(node, row):
    room = 0 
    # here we traverse through tree to find leaf 
    return room 
    

def evaluate(test_db, trained_tree):
    accuracy = 0 
    predicted = []
    actual = []

    actual = split_labels_from_dataset(test_db)

    # passes in test data into tree and tree produces an array of predicted labels 
    for row in test_db:
       predicted.append(predict())

    # pass the array of predicted lables and actual labels into the find accuracy function 
    accuracy = find_accuracy(predicted,actual)
    return accuracy 
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
    
     
     
     



