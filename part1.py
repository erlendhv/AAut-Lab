from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


from scipy import stats
import numpy as np


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PolynomialFeatures


X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')


def data_info(X_train, y_train):
    # Convert the NumPy arrays into a DataFrame for easier analysis
    df = pd.DataFrame(X_train, columns=[
        f'feature_{i}' for i in range(X_train.shape[1])])
    df['target'] = y_train

    # Show summary statistics
    print("Summary Statistics")
    print(df.describe())

    # Calculate correlation matrix
    correlation_matrix = df.corr()

    # Show the correlation between features and target
    print("Correlation Matrix")
    print(correlation_matrix['target'])

    # Create pair plots between features and target
    sns.pairplot(df, vars=[f'feature_{i}' for i in range(
        X_train.shape[1])], hue='target')
    plt.show()

    for i in range(X_train.shape[1]):
        sns.boxplot(x='target', y=f'feature_{i}', data=df)
        plt.title(f'Boxplot for Feature {i} vs Target')
        plt.show()

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Get feature importance
    importances = model.feature_importances_

    print("Feature Importance")
    # Display feature importance
    for i, importance in enumerate(importances):
        print(f'Feature {i}: {importance}')


def outlier_detection(X_train, y_train):
    # Compute the Z-scores of each feature in the dataset
    z_scores = np.abs(stats.zscore(X_train))

    # Set a threshold to identify outliers
    threshold = 3

    # Identify the indices of rows without outliers
    clean_indices = (z_scores < threshold).all(axis=1)

    # Filter out outliers in both X_train and y_train
    X_train_clean = X_train[clean_indices]
    y_train_clean = y_train[clean_indices]

    print(f"Removed {len(X_train) - len(X_train_clean)} outliers.")
    return X_train_clean, y_train_clean


# Function to detect outliers using MAD

def outlier_detection_mad(X_train, y_train):
    def mad_based_outlier(points, threshold=3.5):
        # Median of the data
        median = np.median(points, axis=0)

        # Absolute deviation from the median
        abs_deviation = np.abs(points - median)

        # Median Absolute Deviation
        mad = np.median(abs_deviation, axis=0)

        # Avoid division by zero by adding a small constant (e.g., 1e-9)
        mad = mad + 1e-9

        # Compute the modified Z-score (scaled by a factor of 1.4826)
        modified_z_score = 0.6745 * abs_deviation / mad

        # Identify outliers based on the threshold
        return modified_z_score > threshold

    # Apply the MAD-based outlier detection to each feature in X_train
    outliers = np.any(mad_based_outlier(X_train), axis=1)

    # Filter out outliers
    X_train_clean = X_train[~outliers]
    y_train_clean = y_train[~outliers]

    print(f"Removed {np.sum(outliers)} outliers using MAD.")
    return X_train_clean, y_train_clean


def ridge_L2_regularization(X_train_clean, y_train_clean):
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)

    # Ridge Regression with cross-validation for hyperparameter tuning
    # Increase max_iter to handle convergence issues
    ridge = Ridge(max_iter=5000)
    # Alpha is the regularization strength
    param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
    ridge_cv = GridSearchCV(ridge, param_grid, cv=5)
    ridge_cv.fit(X_train_scaled, y_train_clean)

    # Best alpha value and the corresponding score
    print(f"Best alpha: {ridge_cv.best_params_['alpha']}")
    print(f"Best cross-validated score: {ridge_cv.best_score_}")


def lasso_L1_regularization(X_train_clean, y_train_clean):
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)

    # Lasso Regression with cross-validation for hyperparameter tuning
    # Increase max_iter to handle convergence issues
    lasso = Lasso(max_iter=5000)
    # Alpha is the regularization strength
    param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
    lasso_cv = GridSearchCV(lasso, param_grid, cv=5)
    lasso_cv.fit(X_train_scaled, y_train_clean)

    # Best alpha value and the corresponding score
    print(f"Best alpha: {lasso_cv.best_params_['alpha']}")
    print(f"Best cross-validated score: {lasso_cv.best_score_}")


def multicollinearity_pca(X_train_clean, y_train_clean):
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)

    # Apply PCA to reduce the dimensionality before regression
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Ridge Regression with cross-validation for hyperparameter tuning
    ridge = Ridge(max_iter=5000)
    # Alpha is the regularization strength
    param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
    ridge_cv = GridSearchCV(ridge, param_grid, cv=5)

    # Fit the Ridge model on PCA-transformed data
    ridge_cv.fit(X_train_pca, y_train_clean)

    print(f"Best alpha: {ridge_cv.best_params_['alpha']}")
    print(f"Best cross-validated score: {ridge_cv.best_score_}")


def ridge_L2_with_polynomial(X_train_clean, y_train_clean, degree=2):
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)

    # Generate polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train_scaled)

    # Ridge Regression with cross-validation for hyperparameter tuning
    ridge = Ridge(max_iter=20000)
    # Alpha is the regularization strength
    param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
    ridge_cv = GridSearchCV(ridge, param_grid, cv=5)
    ridge_cv.fit(X_train_poly, y_train_clean)

    # Best alpha value and the corresponding score
    print(f"Best alpha: {ridge_cv.best_params_['alpha']}")
    print(f"Best cross-validated score: {ridge_cv.best_score_}")


def lasso_L1_with_polynomial(X_train_clean, y_train_clean, degree=2):
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)

    # Generate polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train_scaled)

    # Lasso Regression with cross-validation for hyperparameter tuning
    lasso = Lasso(max_iter=20000)
    # Alpha is the regularization strength
    param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
    lasso_cv = GridSearchCV(lasso, param_grid, cv=5)
    lasso_cv.fit(X_train_poly, y_train_clean)

    # Best alpha value and the corresponding score
    print(f"Best alpha: {lasso_cv.best_params_['alpha']}")
    print(f"Best cross-validated score: {lasso_cv.best_score_}")


if __name__ == "__main__":
    data_info(X_train, y_train)
    # Choose outlier detection method (Z-score or MAD)
    # X_train_clean, y_train_clean = outlier_detection(X_train, y_train)
    print("Outlier Detection with Z-score")
    X_train_clean, y_train_clean = outlier_detection_mad(X_train, y_train)

    # Regularization with Ridge and Lasso after outlier removal and scaling
    print("Regularization with Ridge")
    ridge_L2_regularization(X_train_clean, y_train_clean)
    print("Regularization with Lasso")
    lasso_L1_regularization(X_train_clean, y_train_clean)

    # Add polynomial features and apply Ridge and Lasso regularization
    print("Ridge L2 with Polynomial Features")
    ridge_L2_with_polynomial(X_train_clean, y_train_clean, degree=3)
    print("Lasso L1 with Polynomial Features")
    lasso_L1_with_polynomial(X_train_clean, y_train_clean, degree=3)

    # Apply PCA and Ridge
    print("PCA with Ridge")
    multicollinearity_pca(X_train_clean, y_train_clean)
