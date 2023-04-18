import time
import unittest
from numpy import exp
from model_utilities.regression.logistic import NumpyLassoLogisticModel


class TestNumpyLassoLogisticModel(unittest.TestCase):
    def test_NumpyLassoLogisticModel_fit_without_intercepts(self):
        lasso_model_no_intercept = NumpyLassoLogisticModel(alpha=0.0)

        X = [x / 10. - 5. for x in range(10000)]
        Y = [1. / (1. + exp(-1.5 * x - 0.5)) for x in X]
        X = [[x] for x in X]

        start = time.time_ns()
        lasso_model_no_intercept.fit(X, Y, global_seed=True)
        end = time.time_ns()

        print(f"Numpy {(end - start) / 1000000000}")

        self.assertTrue(lasso_model_no_intercept._initialized)
        self.assertAlmostEqual(lasso_model_no_intercept.get_parameters()[0], -1.5, 4)
        self.assertAlmostEqual(lasso_model_no_intercept.get_parameters()[1], -0.5, 4)
