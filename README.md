
# Introduction to Machine Learning - Coursework 1

This project contains code for the first coursework for the Introduction to Machine Learning course. Below we have added instructions to help explain how to run our code.


## Authors

- Daniel Ong
- Chenghong Ren
- Arthika Sivathasan 

## Documentation

Our decision tree can be run by running the decision_tree.py file, which will output data for both our pruned and unpruned decision tree. Below is a directory of our functions, detailing what each function does.

## main()

## decision_tree_learning()

## evaluate()

## prune_tree()

## get_metrics()
- _Parameters of the Function: `test_db`, `trained_tree`_
- _Values Returned: `accuracy`, `conf_matrix`, `recall`, `precision`, `f1_measure`_

- `get_metrics(test_db, trained_tree)` is a function that returns values for Accuracy, Recall, Precision and F1 Measure by taking in the trained tree and the test data we wish to test the decision tree on. 
- Values for Recall, Precision and F1 Measure are retuned within an array where each element of the array corresponds to the labels: Room 1 through to Room 4 respectively. 
- Within our code, we utilise this function to return the metrics for the averaged matrix from the pruned and unpruned tree. 
## draw_tree()

## How to Run Our Code:
To run our code:
- Add your dataset into `wifi_db` folder.
- Open the `decision_tree.py` file.
- Begin by replacing the filepath to the dataset, by reassiging the `filepath` variable found under the `RUN_CODE` section of the `decision_tree.py` file.
- Run the `decision_tree.py` file and you should see the metrics for the pruned and unpruned tree be printed into the terminal. 
- If you wish to print the visualisation of the decision tree, call the `draw_tree()` function in the `RUN_CODE` section and pass in the tree you wish to visualise as the parameter to the function.

