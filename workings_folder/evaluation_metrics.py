
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

def find_accuracy_from_matrix(conf_matrix):
     for i in range(4):
        for j in range(4)):
            if i == j:
                correct_samples = correct_samples + conf_matrix[i, j]

    accuracy = correct_samples/total_samples

    return accuracy


def find_recall(class_num, conf_matrix):

    recall = 0
    total_class_samples = 0

    correct_class_samples = conf_matrix[class_num - 1, class_num - 1 ]

    for sample in actual_labels:
        if sample == class_num:
            total_class_samples += 1

    recall = correct_class_samples / total_class_samples

    return recall


def find_precision(class_num, conf_matrix):

    precision = 0
    total_class_samples = 0

    correct_class_samples = conf_matrix[class_num - 1 , class_num - 1]

    for sample in predicted_labels:
        if sample == class_num:
            total_class_samples += 1

    precision = correct_class_samples / total_class_samples

    return precision


def find_F1(class_num, conf_matrix):

    f_measure = 0
    recall = find_recall(class_num, conf_matrix)
    precision = find_precision(class_num, conf_matrix)

    f_measure = (2*recall*precision)/(precision + recall)

    return f_measure


def get_metrics(avg_conf_matrix):
  
    recall = []
    precision = []
    f1_measure = []
    conf_matrix = np.zeros((4, 4))

    accuracy = find_accuracy_from_matrix(avg_conf_matrix)
  
    for x in range(1, 5):
        recall.append(find_recall(x, avg_conf_matrix ))
        precision.append(find_precision(x, avg_conf_matrix))
        f1_measure.append(find_F1(x, avg_conf_matrix))

    return accuracy, avg_conf_matrix, recall, precision, f1_measure


def main(filename):
    avg_accuracy = 0 
    avg_recall = []
    avg_precision = []
    avg_f1 = []

    cumalative_conf_matrix = np.zeros((4, 4))
    dataset = read_dataset(filename)

    # split data into 10 folds
    split_indices = k_fold_split(10, dataset, rg)

    for k in range(10):
        # pick k as test
        test_indices = split_indices[k]

        # for validation:
        # validate_indicies = split_indicies[k+1] ??
        # train_indices = np.hstack(split_indices[:k+1] + split_indices[k+2:])

        # combine remaining splits as train
        train_indices = np.hstack(split_indices[:k] + split_indices[k+1:])

        trained_tree, depth = decision_tree_learning(train_indices, 0)
        accuracy, conf_matrix = evaluate(test_indices, trained_tree)

        cumalative_conf_matrix += conf_matrix

    average_conf_matrix = cumalative_conf_matrix / 10
    
    avg_accuracy, average_conf_matrix, avg_recall, avg_precision, avg_f1 = get_metrics(average_control_matrix)
    
    return average_conf_matrix
