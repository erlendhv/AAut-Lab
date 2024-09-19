import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Load data
X_train = np.load('X_train.npy')  # Independent variables
y_train = np.load('y_train.npy')  # Dependent variable

# 1. Outlier Removal using MAD (for y_train)
def outlier_detection_mad(y_train, threshold=3.5):
    median = np.median(y_train)
    abs_deviation = np.abs(y_train - median)
    mad = np.median(abs_deviation)
    modified_z_score = 0.6745 * abs_deviation / (mad + 1e-9)
    return modified_z_score < threshold  # Keep values within threshold

# Identify non-outlier samples
clean_indices = outlier_detection_mad(y_train)
X_train_clean = X_train[clean_indices]
y_train_clean = y_train[clean_indices]

# 2. Split data into 80% training and 20% testing (validation)
X_train_80, X_val_20, y_train_80, y_val_20 = train_test_split(X_train_clean, y_train_clean, test_size=0.2, random_state=42)

# 3. Standardize the features (fit on training, transform on both training and validation sets)
scaler = StandardScaler()
X_train_80_scaled = scaler.fit_transform(X_train_80)
X_val_20_scaled = scaler.transform(X_val_20)

# 4. Ridge Regression (L2 regularization)
ridge = Ridge(max_iter=20000)
param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
ridge_cv = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
ridge_cv.fit(X_train_80_scaled, y_train_80)

print(f"Best alpha for Ridge: {ridge_cv.best_params_['alpha']}")
print(f"Best CV score (Ridge): {ridge_cv.best_score_}")

# Test the model on the validation set
ridge_val_pred = ridge_cv.predict(X_val_20_scaled)
ridge_val_mse = np.mean((ridge_val_pred - y_val_20)**2)
print(f"Ridge Validation MSE: {ridge_val_mse}")

# 5. Lasso Regression (L1 regularization)
lasso = Lasso(max_iter=20000)
param_grid_lasso = {'alpha': [0.01, 0.1, 1, 10, 100]}
lasso_cv = GridSearchCV(lasso, param_grid_lasso, cv=5, scoring='neg_mean_squared_error')
lasso_cv.fit(X_train_80_scaled, y_train_80)

print(f"Best alpha for Lasso: {lasso_cv.best_params_['alpha']}")
print(f"Best CV score (Lasso): {lasso_cv.best_score_}")

# Test the model on the validation set
lasso_val_pred = lasso_cv.predict(X_val_20_scaled)
lasso_val_mse = np.mean((lasso_val_pred - y_val_20)**2)
print(f"Lasso Validation MSE: {lasso_val_mse}")

# 6. Make predictions on the provided test set using Ridge (you can use Lasso similarly)
X_test = np.load('X_test.npy')  # Assuming you have a test set
X_test_scaled = scaler.transform(X_test)  # Standardize the test set

y_pred_ridge = ridge_cv.predict(X_test_scaled)
np.save('y_pred_ridge.npy', y_pred_ridge)  # Save Ridge predictions

y_pred_lasso = lasso_cv.predict(X_test_scaled)
np.save('y_pred_lasso.npy', y_pred_lasso)  # Save Lasso predictions
