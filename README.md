# ML From Scratch in Python

Small machine learning implementations written directly in Python and NumPy.

## Focus Areas

- Linear models
- Gradient descent
- Loss functions
- Evaluation metrics
- Reproducible training examples

## Included Modules

- `ml/linear_regression.py` - batch gradient descent linear regression
- `ml/metrics.py` - mean squared error and R-squared
- `examples/train_linear_regression.py` - synthetic training example
- `tests/test_linear_regression.py` - regression and metrics tests

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python examples/train_linear_regression.py
```

## Notes

The goal is to keep the code readable enough to explain line by line while still using tests and repeatable examples.
