import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng


def read_dataset(filepath):

    x = []
    for line in open(filepath):
        if line.strip() != "":  # handle empty rows in file
            row = line.strip().split()
            x.append(list(map(float, row)))

    x = np.array(x)
    return x


def find_info_gain():
    # implement the calculation to find the info gain for a split
    pass


def find_split():
    # declare new variable best_split and best_info_gain
    # go through each possible split
    # calculate the information gain for each split
    # if the information gain is better for the current best_info_gain,
    # reassign best_split to the current split

    pass

def is_leaf():
    pass 

def evaluate(test_db, trained_tree):
    pass 

#________________________________________EVALUATION_METRICS__________________________________________

def create_confusion_matrix(predicted_labels, actual_labels):

    classes = np.unique(actual_labels)

    conf_matrix = np.zeros((len(classes), len(classes)))
  
    for i in range(len(classes)):

        for j in range(len(classes)):
           conf_matrix[i, j] = np.sum((actual_labels == classes[i]) & (predicted_labels == classes[j]))

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
        correct_samples = correct_samples + conf_matrix[i,j]

  accuracy = (correct_samples/total_samples) * 100 # accuracy is given as a percentage 

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

#____________________________________________________BUILDING_MODEL__________________________________________________________

def decision_tree_learning(dataset, depth = 0 ):

    is_leaf = is_leaf()

    if is_leaf == True :
        pass

    else:
        for col_num in range (0,7):
            best_split = find_split()
            left_dataset, right_dataset = split_dataset()
            node = "put smth here"
            left_branch, left_depth = decision_tree_learning(left_dataset, depth + 1)
            right_branch, right_depth = decision_tree_learning(right_dataset, depth + 1)
        
            return (node, max(left_depth, right_depth))



x = read_dataset("wifi_db/clean_dataset.txt")


