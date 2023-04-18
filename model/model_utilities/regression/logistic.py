from functools import partial
from typing import Optional, Sequence
from numpy import array, dot, append, ones, exp, sum, abs
from scipy.optimize import minimize, basinhopping


class NumpyLassoLogisticModel:
    """
    This model_utilities is a logistic REGRESSION - it does not do classification, it fits a logistic curve to data as a regression. This class also allows for an alpha parameter so you can do lasso dimensional reduction. The parameters are the linear parameters (C), then the line's intercept (D), then the magnitude of the curve (A), then the vertical adjustment (B).

    Note that logistic fits have an interesting symmetry in them that needs to be accounted for. If I say my equation is B + A / (1 + exp(x*C + D)), I have the following symmetry (notice that I can apply it twice to return to the original answer!):
        A -> -A
        B -> B + A
        C -> -C
        D -> -D
    """
    def __init__(self,
                 alpha: float = 0.0,
                 tol: float = None):
        self._alpha: float = alpha
        self._tol = tol

        self._parameters: Optional[Sequence[float]] = None
        self._initialized: bool = False

    def _evalute_point(self, X: array, theta: array) -> array:
        """
        Evaluates points of the logistic curve.

        Args:
            x (array): A 2-dimensional array with the number of points and the point dimensions. Assumes an extra value of "1" has been appended to the inputs so the shapes match.
            theta (array): The parameters of the logistic model_utilities. Must be a 1-dimensional array.

        Returns:
            array: The predictions for the points.
        """
        return 1 / (1 + exp(dot(X, theta)))

    def _constrained_lasso(self, theta: array, X: array, Y: array) -> array:
        """
        Evaluates points of the logistic curve.

        Args:
            theta (array): The parameters of the logistic model_utilities. Must be a 1-dimensional array.
            x (array): A 2-dimensional array with the number of points and the point dimensions. Assumes an extra value of "1" has been appended to the inputs so the shapes match.
            y (array): A 1-dimensional array with the observed points.

        Returns:
            array: The predictions for the points.
        """
        return sum((Y - self._evalute_point(X, theta)) ** 2) / 2 / X.shape[0] + sum(self._alpha * abs(theta))

    def _copy_and_add_intercept(self, x: array) -> array:
        return append(x, ones((x.shape[0], 1)), axis=1)

    # BFGS isn't very good on constrained optimization. Nelder-Mead does a much better job.
    def fit(self, X: Sequence[Sequence[float]], Y: Sequence[float], x0: Optional[Sequence[float]] = None,
            method: str = 'Nelder-Mead', global_seed: bool = False):
        return self._internal_fit(X, Y, x0, method, global_seed)

    # BFGS isn't very good on constrained optimization. Nelder-Mead does a much better job.
    def _internal_fit(self, x: Sequence[Sequence[float]], y: Sequence[float], x0: Optional[Sequence[float]] = None,
                      method: str = 'Nelder-Mead', global_seed: bool = False):
        copied_x = self._copy_and_add_intercept(array(x))
        copied_y = array(y)

        fit_fun = partial(self._constrained_lasso, X=copied_x, Y=copied_y)

        x0_internal = x0
        if x0_internal is None:
            x0_internal = array([1.0 / (copied_x.shape[1] - 1)] * (copied_x.shape[1] - 1) + [0.0])

        if global_seed:
            xhat_seed = basinhopping(fit_fun, x0_internal)
            x0_internal = xhat_seed.x
        xhat = minimize(fit_fun, x0_internal, method=method, tol=self._tol)

        self._parameters = xhat.x
        self._initialized = True

    def get_parameters(self) -> Sequence[float]:
        return self._internal_get_parameters()

    def _internal_get_parameters(self) -> Sequence[float]:
        return self._parameters

    def get_initialized(self) -> bool:
        return self._internal_get_initialized()

    def _internal_get_initialized(self) -> bool:
        return self._initialized
