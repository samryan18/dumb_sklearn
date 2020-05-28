from numpy import array, cumsum, matmul, mean, std
from numpy.linalg import eig


def sort_eigenstuff(e: array, v: array) -> tuple:
    """ Utility func Sort eigenvalues and eigenvalues by descending magnitude """
    argsort = e.argsort()[::-1]
    return e[argsort], v[:, argsort]


class PCA:
    def __init__(self, n_components: int, standardize: bool = True):
        """Create PCA.

        1. Centers data about the origin and standardizes (optional).
        2. Solves for covariance matrix: Z^TZ
        3. Finds eigendecomposition and selects n eigenvectors.
        
        Arguments:
            n_components {int} -- Number of principal components to store
        
        Keyword Arguments:
            standardize {bool} -- Whether to standardize the variance of the data (default: {True})
        """
        self.n_components = n_components
        self._is_standardized = standardize
        self._is_solved = False

    def _standardize(self, data: array) -> array:
        """ Could name this function better, center and/or standardize """
        self._mean = mean(data, axis=0)
        if not self._is_standardized:
            # (X-x_bar)
            return data - self._mean
        else:
            # (X-x_bar)/STD
            self._std = std(data, axis=0)
            return (data - self._mean) / self._std

    def _inverse_standardize(self, data: array) -> array:
        """ Reverse the standardize operation """
        if not self._is_standardized:
            # (X+x_bar)
            return data + self._mean
        else:
            # X*STD+x_bar
            return self._std * data + self._mean

    def transform(self, data: array) -> array:
        """ Throw the matrix in our PCA washing machine """
        if self.n_components > data.shape[1]:
            raise ValueError("Too many components!")

        self.check_if_solved()

        return matmul(self._standardize(data), self.v[:, : self.n_components])

    def inverse_transform(self, X_transform: array) -> array:
        """ Description in the name """
        return self._inverse_standardize(
            matmul(X_transform, self.v[:, : self.n_components].T)
        )

    def fit(self, data: array):
        """Fit PCA

        1. Standardize (find Z matrix)
        2. Solve for covariance matrix: matmul(Z^T, Z)
        3. Find eigenstuff of covariance matrix
        4. Sort eigenstuff based on eigenvalue absolute value
        
        Args:
            data: array with shape (n_datapoints, n_features)
        """

        data = self._standardize(data)

        covariance_matrix = matmul(data.T, data)

        self.e, self.v = sort_eigenstuff(*eig(covariance_matrix))
        # by numpy convention:
        # eigenvectors[:,2] to access the third eigenvector

        self._is_solved = True

    def fit_transform(self, data: array) -> array:
        """ Fit... then transform! """
        self.fit(data)
        return self.transform(data)

    def explained_variance(self, pc_num: int) -> float:
        """ Get the explained variance for a particular pc_num """
        self.check_if_solved()
        return (self.e / self.e.sum())[pc_num]

    def cumulative_explained_variance(self, pc_num: int) -> float:
        """ Get the cumulative explained variance up to a particular pc_num """
        self.check_if_solved()
        return cumsum(self.e / self.e.sum())[pc_num]

    def check_if_solved(self) -> bool:
        if not self._is_solved:
            raise ValueError("Model not yet fit to data!")

    @property
    def components(self) -> array:
        """ Get the principal component vectors """
        self.check_if_solved()
        return self.v[:, : self.n_components].T
