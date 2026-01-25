# Required assignment 5.1 Applying the training set–validation set–test set approach

In this notebook, you will use the training and validation sets to identify which model best fits the data. You will be working with the `train-test-split` approach from sklearn.

### **Steps followed in the `train_test_validate` approach**

1. Split the available data into a training set, a validation set and a test set.

2. Fit each model separately on the training set.

3. Evaluate each model separately on the validation set.

4. Choose the model that performs best on the validation set.

5. Estimate the performance of that model on the test set.

6. Train the selected model again using all data.


```python
#Import the necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
```

Use the polynomial function to generate a data set. The function is defined as:
$$
y = 3x^3 - 2x^2 + x + 5
$$


```python
# Example: generate synthetic data (replace with your own X, y)
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 3 * X.squeeze()**3 - 2 * X.squeeze()**2 + X.squeeze() + 5 + np.random.randn(100) * 100
```

### Step 1: Split the data into a training, a validation and a test set

#### **Question 1:** For the synthetic data set, use the `train_test_split` function to split the data into 60 per cent training , 20 per cent validation and 20 per cent test data set.

HINT: Since the `train_test_split()` function can only split the data set into two parts, follow these steps:

- The first split uses 20 per cent of the data for the test data set.

- The second split uses 25 per cent of the previous training set as the test data set.
- Then, from the remaining 80%, split off 25% for validation (which is 25% * 80% = 20% of the total).

- The leftover 60% will be the training set.

- The final proportions of the train:validate:test is 60:20:20.


```python
from sklearn.model_selection import train_test_split

# First split: train_val (80%) and test (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Second split: train (60%) and validation (20%)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42
)

```


```python

```

### Step 2: Fit each model separately on the training data set.

#### **Question 2:** For the synthetic data set, use the `LinearRegression()` model.

HINT: Use the `model.fit()` function to fit the model.


```python
val_errors = {}
models = {}

for degree in range(1, 7):
    poly = PolynomialFeatures(degree=degree)

    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_val_pred = model.predict(X_val_poly)
    val_errors[degree] = mean_squared_error(y_val, y_val_pred)

    models[degree] = (model, poly)

print("Models trained:", models.keys())
```

    Models trained: dict_keys([1, 2, 3, 4, 5, 6])



```python

```


```python

```

#### Step 3: Evaluate each model separately on the validation data set.

#### **Question 3:** Evaluate on the validation set and use the `mean_squared_error()` to compute the error.


```python
###GRADED
for degree in models:
    model, poly = models[degree]

    X_val_poly = poly.transform(X_val)
    y_val_pred = model.predict(X_val_poly)

    mse = mean_squared_error(y_val, y_val_pred)
    val_errors[degree] = mse

    print(f"Degree {degree}: Validation MSE = {mse:.2f}")
```

    Degree 1: Validation MSE = 161492.22
    Degree 2: Validation MSE = 7467.75
    Degree 3: Validation MSE = 8567.48
    Degree 4: Validation MSE = 8117.44
    Degree 5: Validation MSE = 8036.56
    Degree 6: Validation MSE = 8060.76



```python

```

#### Step 4: Choose the model that performs best on the validation set.

#### **Question 4**: Choose the best model that has the mininum error and print the degree of that model.

HINT: Use `key=val_errors.get` to use the dictionary values (the validation errors) for comparison, not the keys themselves.


```python
###GRADED
best_degree = min(val_errors, key=val_errors.get)
print(f"Best polynomial degree: {best_degree}")
```

    Best polynomial degree: 2



```python

```

#### Step 5: Estimate the performance of that model on the test set.

#### **Question 5**: Estimate the performance on the test set.


```python
###GRADED
best_model, best_poly = models[best_degree]

# Transform test data
X_test_poly = best_poly.transform(X_test)

# Predict on test set
y_test_pred = best_model.predict(X_test_poly)

# Compute test MSE
test_mse = mean_squared_error(y_test, y_test_pred)

print(f"Test MSE for best model (degree {best_degree}): {test_mse:.2f}")
```

    Test MSE for best model (degree 2): 7816.15



```python
    
```

#### Step 6: Retrain the best model on all available data.


```python
best_poly_full = PolynomialFeatures(degree=best_degree)
X_full_poly = best_poly_full.fit_transform(X)
final_model = LinearRegression()
final_model.fit(X_full_poly, y)
print(f"Retrained best polynomial model (degree {best_degree}) on all data.")
```

    Retrained best polynomial model (degree 2) on all data.



```python

```
