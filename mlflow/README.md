# MLFlow Utilities

This package contains wrapper APIs for MLFlow that assist in tracking semantic versioning of models and enabling automation.

This package modifies MLFlow's naming conventions a touch. Remember, a "Run" is the single output of a training process being run. A "Model" is used as the actual logic that gets saved and associated to a run (not to be confused with the vernacular "model", which is an "Experiment"). If there are multiple statistical models for a single model project, those are saved in the same Experiment as different submodels.

The model status framework by way of Enum is sufficient to handle every operational structure except online learning (which doesn't scale and isn't stable anyway). "Disabled" is when a model is completely disabled - no longer in use. "New" is a transitory state that indicates that the model was trained but has not been evaluated for production use yet. "Active" are models that are in use in production making predictions. "Canary" are models that you want to make predictions with but don't want to use the results in production yet.

Package structure with usage is below:

## mlflow_utilities

### mlflow_api

| function name         | arguments                                                                                                                            | description                                                                                                                                                                    |
|-----------------------|--------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| save_model            | model, model_version, mlflow_subpackage, experiment_id, experiment_name, submodel_name, immutable_metadata, mutable_metadata         | Saves a model using MLFlow, with an added semantic versioning structure on top.                                                                                                |
| change_status         | run_id, new_active_state, new_test_fraction                                                                                          | Update a run's ability to run within the semver framework.                                                                                                                     |
| enable_run            | run_id                                                                                                                               | Enables a run by setting it to be the active model with 100% of the A/B fraction.                                                                                              |
| disable_run           | run_id                                                                                                                               | Disables a run.                                                                                                                                                                |
| canary_run            | run_id                                                                                                                               | Takes a run and sets it to the Canary status.                                                                                                                                  |
| update_active_runs    | test_fraction_by_run, model_version, experiment_id, experiment_name, submodel_name, extra_immutable_metadata, extra_mutable_metadata | Updates which runs are going to be active. Ignores disabled models. Will disable any active and canary runs which are not present. Sets the A/B test fractions in the process. |
| change_test_fractions | run_id                                                                                                                               | Updates A/B test fraction for all runs of the provided type.                                                                                                                   |
| list_runs             | model_version, experiment_id, experiment_name, active_state, submodel_name, extra_immutable_metadata, extra_mutable_metadata         | Method to list runs within the framework.                                                                                                                                      |
| list_models           | model_version, experiment_id, experiment_name, active_state, submodel_name, extra_immutable_metadata, extra_mutable_metadata         | Method to list models within the framework. This will be the hook to retrieve models in your production service or application.                                                |
