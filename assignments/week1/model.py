import numpy as np


class LinearRegression:
    """
    A linear regression model that uses the normal closed form equation to fit the model.
    """

    w: np.ndarray
    b: float

    def __init__(self):
        """
        Initialize w: the coefficients and b: the intercept.
        """
        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model to the input and output data using linear algebra closed form.

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The output data.

        Returns:
            None
        """
        X_new = np.hstack((np.ones((X.shape[0], 1)), X))
        self.w = np.linalg.inv(X_new.T @ X_new) @ X_new.T @ y
        self.b = self.w[0]
        self.w = self.w[1:]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.
        """
        X_new = np.hstack((np.ones((X.shape[0], 1)), X))
        return X_new @ np.hstack((self.b, self.w))


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """
        Fit the model to the given input and output data.

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The output data.
            lr (float): The learning rate.
            epochs (int): The number of epochs for training.
        """
        X_new = np.hstack((np.ones((X.shape[0], 1)), X))
        self.w = np.random.randn(X_new.shape[1])
        self.b = np.random.randn()
        for i in range(epochs):
            y_pred = X_new @ self.w + self.b
            errors = y - y_pred

            self.b += lr * np.square(errors).mean()
            self.w += lr * (X_new.T @ errors)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        X_new = np.hstack((np.ones((X.shape[0], 1)), X))
        return X_new @ self.w + self.b
