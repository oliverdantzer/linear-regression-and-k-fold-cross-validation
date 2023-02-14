# Linear regression and k-fold cross-validation
## Data source
https://www.kaggle.com/datasets/elmoallistair/commodity-prices-19602021

## Objective
Find variable best modelled by the other variables using linear regression. Then use k-fold cross-validation to test the performance of using linear regression to predict the variable in question.

## Methods
Data values ranged from scale of 10^-2 to 10^1, therefore data was standardized before linear regression was performed in order to prevent biases toward data with greater values.

For each column of data:

 - Column is considered as dependent variable
 - Remainder of columns considered as independent variables, gather in matrix
 - Perform linear regression to find model for dependent variable
 - Apply model to matrix of independent variables to get prediction for dependent variable
 - Record RMS error between our prediction and observation for the dependent variable.

The variable with the lowest RMS error is the variable best modelled by the others.

A 9-fold cross-validation was then performed with the best-modelled variable as the dependent variable.

## Results
The index of the variable with the lowest RMS error was 28, corresponding to the commodity soybean.

![This is an image](https://github.com/oliverdantzer/linear-regression-and-k-fold-cross-validation/blob/main/Figure%201.png?raw=true)

Using soybean as the dependent variable for a 9-fold cross-validation, the training RMS error was 50.8251% less than validation RMS error, indicating the model was overfitting the the training data.

## Running
In MATLAB, with Statistics and Machine Learning Toolbox add-on installed, run ```run.m```
