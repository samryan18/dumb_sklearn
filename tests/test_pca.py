import numpy as np
import pytest
from dumb_sklearn import PCA


def test_near_perfect_inversion():
    x = np.array([[1, 0, 2], [0, 1, 0], [0, 1, 0], [0, 3, 1]])
    pca = PCA(n_components=3)
    x2 = pca.inverse_transform(pca.fit_transform(x))
    assert abs(np.sum(x - x2)) < 0.00001, (
        "Full rank eigenbasis should invert near perfectly "
        "up to floating point error"
    )


def test_early_inversion_raises():
    pca = PCA(n_components=3)

    with pytest.raises(ValueError):
        pca.transform(np.random.random((2, 2)))


def test_early_property_access_raises():
    pca = PCA(n_components=3)

    with pytest.raises(ValueError):
        pca.components

    with pytest.raises(ValueError):
        pca.explained_variance(1)

    with pytest.raises(ValueError):
        pca.cumulative_explained_variance(1)
