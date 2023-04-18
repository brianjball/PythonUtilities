# Model Utilities

This package contains some utilities that are useful for Data Science activities and that don't require additional packages over numpy and scipy.

Package structure with usage is below:

## model_utilities

### evaluation

#### class_cutoff

| function name           | arguments                                                                                                     | description                                                                                                                                                                                                                |
|-------------------------|---------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| find_cutoff_sorted      | sorted_test_pairs, cutoff_method, false_positive_versus_negative_importance, use_rates_over_counts            | Finds the optimal cutoff for a two-class classification when provided some inputs about how the optimization should be run. Data is provided as a target actual 0/1 and probability pair sorted descending by probability. |
| find_cutoff             | actual, predicted, cutoff_method, false_positive_versus_negative_importance, use_rates_over_counts            | Finds the optimal cutoff for a two-class classification when provided some inputs about how the optimization should be run. Actual and predicted lists must be the same size and in the same order.                        |
| find_multiclass_cutoffs | actual, predicted, class_map, cutoff_method, false_positive_versus_negative_importance, use_rates_over_counts | Finds the optimal cutoff for a multi-class classification when provided some inputs about how the optimization should be run. Outputs are not guaranteed to have exact coverage.                                           |

### regression

#### logistic

Provides a class NumpyLassoLogisticModel, which fits a logistic shape to data. This model is ideal for a system with 2 specific conditions: your predicted values must be bounded above and below, and the target is monotonic with the variables. API matches that of models in numpy/scipy.
