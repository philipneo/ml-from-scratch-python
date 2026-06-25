import numpy as np

from ml.linear_regression import LinearRegressionGD
from ml.metrics import mean_squared_error, r2_score


def test_linear_regression_learns_simple_line():
    x = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    y = 2.0 * x + 1.0

    model = LinearRegressionGD(learning_rate=0.03, epochs=2_500).fit(x, y)
    predictions = model.predict(x)

    assert mean_squared_error(y, predictions) < 0.01
    assert r2_score(y, predictions) > 0.99


def test_predict_requires_fit_first():
    model = LinearRegressionGD()

    try:
        model.predict(np.array([1, 2, 3], dtype=float))
    except RuntimeError as error:
        assert "fit" in str(error)
    else:
        raise AssertionError("predict should fail before fit")
