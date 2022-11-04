# Introduction to Machine Learning - Coursework 1

This project contains code for the first coursework for the Introduction to Machine Learning course. Below we have added instructions to help explain how to run our code.

## TO DO LIST
- In the report: 
  - Add the depth values in the DEPTH ANALYSIS section
- In readme.md
  - Add information for draw_tree
- In the code:
  - Output the depth of the pruned tree

## Authors

- Daniel Ong
- Chenghong Ren
- Arthika Sivathasan 

## Documentation

Our decision tree can be run by running the decision_tree.py file, which will output data for both our pruned and unpruned decision tree. Below is a directory of our functions, detailing what each function does.

## main()
- _Parameters of the Function: `filename`_
- _Values Returned: All the averaged clasification metrics and confusion matrix for both the pruned and unpruned tree._
- The purpose of `main(filename)` is to tie together all the functions by carrying out the 10-fold cross validation and gathering the averaged classification metrics and confusion matrix after creating the pruned and unpruned tree. 

## decision_tree_learning()
- _Parameters of the Function: `training_dataset`, `depth`_
- _Values Returned: `curr`, `max(l_depth, r_depth)`_
- `decision_tree_learning(training_dataset, depth)` works by taking in a dataset and by recursively builiding a tree.
- The `depth` variable is used to keep track of how deep the tree is and the function returns the total depth and the root node of the decision tree.

## evaluate()
- _Parameters of the Function: `test_db`, `trained_tree`_
- _Values Returned: `accuracy`, `conf_matrix`_
- `evaluate(test_db, trained_tree)` is a function that takes in the test dataset and the trained tree. 
- Using the `eval_helper()` function, the `evaluate()` function generates an array of predicted labels for the test data.
- The function then compares the predicted labels and actual labels of the test dataset to produce the accuracy and confusion matrix of the test

## prune_tree()
- _Parameters of the Function:`validate_db`, `train_db`, `tree`, `root`, `pruned`_
- _Values Returned:N/A_
- `prune_tree(validate_db, train_db, tree, root, pruned)` works by intially traversing to the bottom of the decison tree and by working bottom up.
- The function removes the bottom-most node  of the tree and  then checks if perfomance of the tree has improved using the validation set. 
- If the perfomance of tree has improved by removing the node, the function will keep the new version of the tree. If not, the function will replace the node back again. 

## get_metrics()
- _Parameters of the Function: `test_db`, `trained_tree`_
- _Values Returned: `accuracy`, `conf_matrix`, `recall`, `precision`, `f1_measure`_
- `get_metrics(test_db, trained_tree)` is a function that returns values for Accuracy, Recall, Precision and F1 Measure by taking in the trained tree and the test data we wish to test the decision tree on. 
- Values for Recall, Precision and F1 Measure are retuned within an array where each element of the array corresponds to the labels: Room 1 through to Room 4 respectively. 
- Within our code, we utilise this function to return the metrics for the averaged matrix from the pruned and unpruned tree. 

## draw_tree()
- _Parameters of the Function: `node`, `x`, `y`, `width`_
- _Values Returned: N/A_
- The `draw_tree(node, x, y, width)` functions works in conjunction to other helper functions: 
  - `draw_branch(attribute, value, x, y)`
  - `draw_leaf(label, x, y)`
  - `draw_line(x1, y1, x2, y2)`
  - `draw_text(s, x, y)`
- The `draw_tree(node, x, y, width)` works by drawing the tree onto a graph and therefore takes in the parameters:
  - `x`, The x value where we begin to draw the tree. Intitally set this to 0.
  - `y`, The y value where we begin to draw the tree. Intitally set this to 0.
  - `width`, The width is how wide we want to draw the tree.
  - `node`, The current node of the tree we are drawing. Intially set this to the root of the tree.
- The way `draw_tree(node, x, y, width)` works is by traversing the tree and identifying the node we're at. It then calls the appropriate helper functions to draw the node. 

## How to Run Our Code:
To run our code:
- Add your dataset into `wifi_db` folder.
- Open the `decision_tree.py` file.
- Begin by replacing the filepath to the dataset, by reassiging the `filepath` variable found under the `RUN_CODE` section of the `decision_tree.py` file.
- Run the `decision_tree.py` file and you should see the classification metrics and confusion matrix for the pruned and unpruned tree, printed in the terminal. 
- If you wish to print the visualisation of the decision tree, call the `draw_tree()` function in the `RUN_CODE` section and pass in the corresponding values, detailed in the directory of functions above.

