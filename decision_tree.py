
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng

seed = 60012
rg = default_rng(seed)
LABEL_COL = 7


# ____________________________BUILDING TREE_________________________________

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
    return array[array[:, column].argsort()]


def find_split(array):
    array = sort_column(array, 0)
    highest_gain = 0
    l_dataset = array[:1]
    r_dataset = array[1:]
    split = array[1][0]
    b_column = 0
    for column in range(len(array[0])-1):
        array = sort_column(array, column)
        for element in range(len(array)):
            if (element == 0 or array[element][column] != array[element - 1]
                    [column]):
                curr_gain = information_gain(array, element)
                if (curr_gain > highest_gain):
                    highest_gain = curr_gain
                    l_dataset = array[:element]
                    r_dataset = array[element:]
                    split = (array[element][column])
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
    if (training_dataset.ndim == 1):
        curr.value = training_dataset[LABEL_COL]
        return (curr, depth)
    if (same_labels(training_dataset)):
        curr.value = training_dataset[0][LABEL_COL]
        return (curr, depth)
    l_dataset, r_dataset, curr.value, curr.attribute = \
        find_split(training_dataset)
    curr.left, l_depth = decision_tree_learning(l_dataset, depth + 1)
    curr.right, r_depth = decision_tree_learning(r_dataset, depth + 1)
    return curr, max(l_depth, r_depth)


# ____________________________DRAWING TREE_________________________________

DEPTH = 10


def draw_tree(node, x, y, width):
    if node.attribute is None:
        draw_leaf(node.value, x, y)
        return
    if width < 1:
        width = 1
    draw_branch(node.attribute, node.value, x, y)
    xl = x - width / 2
    yl = y - DEPTH
    xr = x + width / 2
    yr = y - DEPTH
    draw_line(x, y, xl, yl)
    draw_tree(node.left, xl, yl, width / 2)
    draw_line(x, y, xr, yr)
    draw_tree(node.right, xr, yr, width / 2)


def draw_branch(attribute, value, x, y):
    feature = 'x' + str(int(attribute))
    s = f'{feature} < {value}'
    draw_text(s, x, y)


def draw_leaf(label, x, y):
    draw_text(str(label), x, y)


def draw_line(x1, y1, x2, y2):
    plt.plot([x1, x2], [y1, y2], color='black')


def draw_text(s, x, y):
    plt.text(x, y, s, ha='center', va='center',
             bbox=dict(facecolor='white', pad=5.0))


# ____________________________PRUNING FUNCTIONS________________________________

def prune_tree(validate_db, train_db, tree, root, pruned):
    if tree.attribute is None:
        return

    col = tree.attribute
    split_point = tree.value

    acc, conf = evaluate(validate_db, root)

    sorted_db = sort_column(train_db, col)

    for i in range(len(sorted_db)):
        if sorted_db[i][col] == split_point:
            l_dataset = sorted_db[:i]
            r_dataset = sorted_db[i:]
            break

    if (tree.left.attribute is None and tree.right.attribute is None):
        tree.attribute = None
        if (len(l_dataset) > len(r_dataset)):
            tree.value = l_dataset[0][LABEL_COL]
        else:
            tree.value = r_dataset[0][LABEL_COL]
        n_acc, n_conf = evaluate(validate_db, root)
        if (n_acc > acc):
            del tree.left, tree.right
            pruned[0] = True
            return
        else:
            tree.attribute = col
            tree.value = split_point

    if (tree.left is not None):
        prune_tree(validate_db, l_dataset, tree.left, root, pruned)

    if (tree.right is not None):
        prune_tree(validate_db, r_dataset, tree.right, root, pruned)


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
        predicted_labels.append(eval_helper(data, trained_tree))
        actual_labels.append(data[LABEL_COL])

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

    correct_class_samples = conf_matrix[class_num - 1, class_num - 1]

    for sample in actual_labels:
        if sample == class_num:
            total_class_samples += 1

    recall = correct_class_samples / total_class_samples

    return recall


def find_precision(class_num, predicted_labels, actual_labels):

    precision = 0
    total_class_samples = 0
    conf_matrix = create_confusion_matrix(predicted_labels, actual_labels)

    correct_class_samples = conf_matrix[class_num - 1, class_num - 1]

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


def get_metrics(test_db, trained_tree):
    predicted_labels = []
    actual_labels = []
    recall = []
    precision = []
    f1_measure = []

    for data in test_db:
        # passes in test data into tree and tree produces an array of predicted
        # labels
        predicted_labels.append(eval_helper(data, trained_tree))
        actual_labels.append(data[LABEL_COL])

    # pass the array of predicted labels and actual labels into the find
    conf_matrix = create_confusion_matrix(predicted_labels, actual_labels)

    accuracy = find_accuracy(predicted_labels, actual_labels)

    for x in range(1, 5):
        recall.append(find_recall(x, predicted_labels, actual_labels))
        precision.append(find_precision(x, predicted_labels, actual_labels))
        f1_measure.append(find_F1(x, predicted_labels, actual_labels))

    return accuracy, conf_matrix, recall, precision, f1_measure


def flatten(list):
    return [item for sublist in list for item in sublist]


def main(filename):

    cumalative_conf_matrix_pruned = np.zeros((4, 4))
    cumalative_conf_matrix_unpruned = np.zeros((4, 4))
    split_cumalative_conf_matrix_pruned = np.zeros((4, 4))

    accuracy_unpruned = 0
    recall_unpruned = []
    precision_unpruned = []
    f1_measure_unpruned = []

    accuracy_pruned = 0
    recall_pruned = []
    precision_pruned = []
    f1_measure_pruned = []

    dataset = read_dataset(filename)

    # split data into 10 folds
    split_indices = k_fold_split(10, dataset, rg)

    for k in range(10):
        # pick k as test
        test_indices = split_indices[k]
        if k == 9:
            validate_indices = split_indices[0]
        else:
            validate_indices = split_indices[k+1]

        # combine remaining splits as train
        train_indices = None
        for i in range(10):
            if i == k or i == k + 1 or (i == 0 and k == 9):
                continue
            if train_indices is None:
                train_indices = split_indices[i]
                continue
            train_indices = np.concatenate((train_indices, split_indices[i]),
                                           axis=0)

        trained_tree, depth = decision_tree_learning(train_indices, 0)
        accuracy_unpruned, conf_matrix, recall_unpruned, precision_unpruned, f1_measure_unpruned = get_metrics(test_indices, trained_tree)

        pruned = [True]
        while pruned[0]:
            pruned = [False]
            prune_tree(validate_indices, train_indices, trained_tree,
                       trained_tree, pruned)

        combined_validate_indices = np.concatenate((validate_indices, train_indices),axis = 0)
        
        accuracy_pruned, pruned_conf_matrix, recall_pruned, precision_pruned, f1_measure_pruned = get_metrics(combined_validate_indices, trained_tree)
        
        cumalative_conf_matrix_pruned += pruned_conf_matrix
        
        cumalative_conf_matrix_unpruned += conf_matrix

    # calculate averaged matrix for both pruned and unpruned data
    average_conf_matrix_unpruned = cumalative_conf_matrix_unpruned / 10
    average_conf_matrix_pruned = cumalative_conf_matrix_pruned / 10

    # get classification metrics for each averaged matrix

    # accuracy_unpruned, average_conf_matrix_unpruned, recall_unpruned, \
    #     precision_unpruned, f1_measure_unpruned = \
    #     get_metrics(test_indices, average_conf_matrix_unpruned)

    # accuracy_pruned, average_conf_matrix_pruned, recall_pruned, \
    #     precision_pruned, f1_measure_pruned = \
    #     get_metrics(test_indices, average_conf_matrix_pruned)

    return accuracy_unpruned, average_conf_matrix_unpruned, recall_unpruned, \
        precision_unpruned, f1_measure_unpruned, accuracy_pruned, \
        average_conf_matrix_pruned, recall_pruned, precision_pruned, \
        f1_measure_pruned


# __________________________________RUN CODE_______________________________

filename = "wifi_db/clean_dataset.txt"
"""
dataset = read_dataset(filename)
tree, depth = decision_tree_learning(dataset, 0)
draw_tree(tree, 0, 0, 10)
plt.show()
accuracy, conf_matrix = evaluate(dataset, tree)
print(accuracy)

"""
accuracy_unpruned, average_conf_matrix_unpruned, recall_unpruned, \
    precision_unpruned, f1_measure_unpruned, accuracy_pruned, \
    average_conf_matrix_pruned, recall_pruned, precision_pruned, \
    f1_measure_pruned = main(filename)

print("Confusion Matrix and Metrics for Unpruned Tree")
print("Confusion Matrix: ")
print(average_conf_matrix_unpruned)
print("Accuracy: ")
print(accuracy_unpruned)
print("Recall: ")
print(recall_unpruned)
print("Precision: ")
print(precision_unpruned)
print("F1 Measure: ")
print(f1_measure_unpruned)


print("Confusion Matrix and Metrics for Pruned Tree")
print("Confusion Matrix: ")
print(average_conf_matrix_pruned)
print("Accuracy: ")
print(accuracy_pruned)
print("Recall: ")
print(recall_pruned)
print("Precision: ")
print(precision_pruned)
print("F1 Measure: ")
print(f1_measure_pruned)


"""
filename = "wifi_db/noisy_dataset.txt"
matrix = main(filename)
print(matrix)

"""
