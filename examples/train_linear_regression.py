from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ml.linear_regression import LinearRegressionGD
from ml.metrics import mean_squared_error, r2_score

rng = np.random.default_rng(7)
x = np.linspace(0, 10, 80)
noise = rng.normal(0, 1.5, size=x.shape)
y = 3.2 * x + 4.5 + noise

model = LinearRegressionGD(learning_rate=0.01, epochs=2_000).fit(x, y)
predictions = model.predict(x)

print("Learned weight:", round(float(model.weights[0]), 3))
print("Learned bias:", round(model.bias, 3))
print("MSE:", round(mean_squared_error(y, predictions), 3))
print("R2:", round(r2_score(y, predictions), 3))
