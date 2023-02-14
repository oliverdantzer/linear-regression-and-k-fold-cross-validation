# Linear regression and k-fold cross-validation
## Data source
https://www.kaggle.com/datasets/elmoallistair/commodity-prices-19602021

## Objective
Find variable best modelled by the other variables using linear regression. Then use k-fold cross-validation to test the performance of using linear regression to predict the variable in question.

## Methods
Data values ranged from scale of 10^-2 to 10^1, therefore data was standardized before linear regression was performed in order to prevent biased toward data with greater values.

For each column of data:

 - Column is considered as dependent variable
 - Remainder of columns considered as independent variables, gather in matrix
 - Perform linear regression to find model for dependent variable
 - Apply model to matrix of independent variables to get prediction for dependent variable
 - Record RMS error between our prediction and observation for the dependent variable.

The variable with the lowest RMS error is the variable best modelled by the others.
