# Label Noise Correction Methods

This Python package provides an implementation of label noise correction algorithms proposed in the literature. These algorithms aim to mitigate the effects of label noise in supervised learning tasks by correcting the noisy labels. The methods were implemented for binary classification tasks.

## Installation

You can install the package using `pip`:

```shell
pip install label_noise_correction
```

## Algorithms

The package currently includes the following label noise correction algorithms:

- **Bayesian Entropy Noise Correction** (BE) [1]
- **Polishing Labels** (PL) [2]
- **Self-Training Correction** (STC) [2]
- **Clustering-Based Correction** (CC) [2]
- **Ordering-Based Noise Correction** (OBNC) [3]
- **Hybrid Label Noise Correction** (HLNC) [4]
- **Fair Ordering-Based Noise Correction** (Fair-OBNC) [5]

## Usage

Here's an example of how to use the package to apply label noise correction:

```python
from label_noise_correction import AlgorithmA
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Introduce label noise to create a noisy dataset
y_noisy = y_train.copy()
for i in y_noisy.index:
    if random.random() < 0.1:
        y_noisy.loc[i] = 1 - y_noisy.loc[i]

# Apply label noise correction
lnc = PolishingLabels(LogisticRegression, 10)
y_corrected = lnc.correct(X_train, y_noisy)

# Train models on the noisy and corrected labels
model = LogisticRegression()
model.fit(X_train, y_noisy)
y_pred_noisy = model.predict(X_test)

model.fit(X_train, y_corrected)
y_pred_corrected = model.predict(X_test)

# Evaluate accuracy before and after correction
accuracy = accuracy_score(y_test, y_pred_noisy)
print("Accuracy before label noise correction:", accuracy)

accuracy = accuracy_score(y_test, y_pred_corrected)
print("Accuracy after label noise correction:", accuracy)
```

## References

1. Sun, Jiang-wen, et al. "Identifying and correcting mislabeled training instances." Future generation communication and networking (FGCN 2007). Vol. 1. IEEE, 2007.
2. Nicholson, Bryce, et al. "Label noise correction methods." 2015 IEEE International Conference on Data Science and Advanced Analytics (DSAA). IEEE, 2015.
3. Feng, Wei, and Samia Boukir. "Class noise removal and correction for image classification using ensemble margin." 2015 IEEE International Conference on Image Processing (ICIP). IEEE, 2015.
4. Xu, Jiwei, Yun Yang, and Po Yang. "Hybrid label noise correction algorithm for medical auxiliary diagnosis." 2020 IEEE 18th International Conference on Industrial Informatics (INDIN). Vol. 1. IEEE, 2020.
5. 

## Contributing

Contributions to this package are welcome! If you have any bug reports, feature requests, or would like to contribute with code improvements, please submit an issue or a pull request on the GitHub repository.

## License

This package is distributed under the [MIT License](https://opensource.org/licenses/MIT).
```

Feel free to modify and expand upon this README.md template according to your specific package and the algorithms you implement.
