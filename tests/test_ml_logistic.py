from __future__ import annotations

import numpy as np

from src.ml.logistic import LogisticRegressionGD


def test_logistic_regression_fits_simple_signal() -> None:
    x = np.array(
        [
            [0.0],
            [0.5],
            [1.0],
            [1.5],
            [2.0],
            [2.5],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)
    model = LogisticRegressionGD(learning_rate=0.1, epochs=200, batch_size=3, seed=7).fit(x, y)
    probs = model.predict_proba(x)[:, 1]
    assert probs[0] < probs[-1]
    assert probs[1] < 0.5
    assert probs[-2] > 0.5

