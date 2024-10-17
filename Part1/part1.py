from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from sklearn.linear_model import HuberRegressor, RANSACRegressor, Ridge, TheilSenRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor


X_train = np.load('Part1/X_train.npy')
y_train = np.load('Part1/y_train.npy')
X_test = np.load('Part1/X_test.npy')


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

# fair to assume close to normal distribution due to the sample size?


def z_outlier_detection(X_train, y_train):
    # Compute the Z-scores of each feature in the dataset
    z_scores = np.abs(stats.zscore(y_train))

    # Identify the indices of rows without outliers
    threshold = 0.9  # removes about 25%
    clean_indices = (abs(z_scores) < threshold)

    # Filter out outliers in both X_train and y_train
    X_train_clean = X_train[clean_indices]
    y_train_clean = y_train[clean_indices]

    print(f"Removed {len(X_train) - len(X_train_clean)} outliers.")
    return X_train_clean, y_train_clean


def local_outlier_factor(X_train, y_train):
    lof = LocalOutlierFactor(n_neighbors=5, contamination=0.1)
    y_reshape = y_train.reshape(-1, 1)
    outliers = lof.fit_predict(y_reshape)
    negative_outlier_factors = lof.negative_outlier_factor_
    threshold = -1.22
    inlier_mask = negative_outlier_factors > threshold
    X_pruned = X_train[inlier_mask]
    y_pruned = y_train[inlier_mask]
    print("Number of outliers detected:", len(y_train) - len(y_pruned))
    return X_pruned, y_pruned


def iqr_outlier_removal(X_train, y_train):
    Q1 = np.percentile(y_train, 25)
    Q3 = np.percentile(y_train, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    inlier_mask = (y_train >= lower_bound) & (y_train <= upper_bound)
    X_pruned = X_train[inlier_mask]
    y_pruned = y_train[inlier_mask]
    print("Number of outliers detected:", len(y_train) - len(y_pruned))
    return X_pruned, y_pruned


def outlier_detection_mad(X_train, y_train):
    def mad_based_outlier(points, threshold=1.5):  # 1.5 removes 51 outliers
        # Median of the data
        median = np.median(points)
        abs_deviation = np.abs(points - median)
        mad = np.median(abs_deviation)
        # avoid division by zero by adding a small constant (e.g., 1e-9)
        mad = mad + 1e-9
        # Compute the modified Z-score (scaled by a factor of 1.4826)
        modified_z_score = 0.6745 * abs_deviation / mad

        # Identify outliers based on the threshold
        return abs(modified_z_score) < threshold

    # Prune training data based on outlier detection
    clean_indices = mad_based_outlier(y_train)
    X_train_clean = X_train[clean_indices]
    y_train_clean = y_train[clean_indices]
    print(f"Removed {200 - np.sum(clean_indices)} outliers using MAD.")

    return X_train_clean, y_train_clean


def ridge_L2_regularization(X_train_clean, y_train_clean):
    # Standardize the features
    scaler = StandardScaler()
    # scaler = RobustScaler()
    # scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)

    # Search for best params using grid search
    ridge = Ridge()
    param_grid = {'alpha': [0.1, 0.2, 0.25, 0.3, 0.5, 1, 10, 100],
                  'max_iter': [250, 400, 500, 600, 750, 1000, 1500, 5000, 10000]}
    ridge_cv = GridSearchCV(ridge, param_grid, cv=5)
    ridge_cv.fit(X_train_scaled, y_train_clean)

    # Best alpha value and the corresponding score
    print(f"Best alpha: {ridge_cv.best_params_['alpha']}")
    print(f"Best cross-validated score: {ridge_cv.best_score_}")

    return ridge_cv.best_params_


def lasso_L1_regularization(X_train_clean, y_train_clean):
    # Standardize the features
    scaler = StandardScaler()
    # scaler = RobustScaler()
    # scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)

    # Search for best params using grid search
    lasso = Lasso()
    param_grid = {'alpha': [0.1, 0.2, 0.25, 0.3, 0.5, 1, 10, 100],
                  'max_iter': [250, 400, 500, 600, 750, 1000, 1500, 5000, 10000]}
    lasso_cv = GridSearchCV(lasso, param_grid, cv=5)
    lasso_cv.fit(X_train_scaled, y_train_clean)

    # Best alpha value and the corresponding score
    print(f"Best alpha: {lasso_cv.best_params_['alpha']}")
    print(f"Best cross-validated score: {lasso_cv.best_score_}")

    return lasso_cv.best_params_


def huber_regression(X_train_clean, y_train_clean):
    # Standardize the features
    scaler = StandardScaler()
    # scaler = RobustScaler()
    # scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)

    huber = HuberRegressor()

    param_grid = {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 10.0],
        'epsilon': [1.1, 1.2, 1.35, 1.5, 1.75, 2.0]
    }

    # Perform Grid Search to find the best parameters
    huber_cv = GridSearchCV(huber, param_grid, cv=5)
    huber_cv.fit(X_train_scaled, y_train_clean)

    # Best alpha and epsilon
    print(f"Best alpha (Huber): {huber_cv.best_params_['alpha']}")
    print(f"Best epsilon (Huber): {huber_cv.best_params_['epsilon']}")
    print(f"Best cross-validated score (Huber): {huber_cv.best_score_}")

    return huber_cv.best_params_


def huber_regression_with_validation(X_train_clean, y_train_clean, X_val, y_val):
    # Standardize the features
    scaler = StandardScaler()
    # scaler = RobustScaler()
    # scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_val_scaled = scaler.transform(X_val)

    huber = HuberRegressor()

    param_grid = {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 10.0],
        'epsilon': [1.1, 1.2, 1.35, 1.5, 1.75, 2.0]
    }

    huber_cv = GridSearchCV(huber, param_grid, cv=5)
    huber_cv.fit(X_train_scaled, y_train_clean)

    print(f"Best alpha (Huber): {huber_cv.best_params_['alpha']}")
    print(f"Best epsilon (Huber): {huber_cv.best_params_['epsilon']}")
    print(f"Best cross-validated score (Huber): {huber_cv.best_score_}")

    y_val_pred = huber_cv.predict(X_val_scaled)
    print(
        f"Huber MSE on Validation Set: {mean_squared_error(y_val, y_val_pred)}")
    print(f"Huber R2 on Validation Set: {r2_score(y_val, y_val_pred)}")

    plot_residuals(y_val, y_val_pred, title="huber residual plot")

    return huber_cv.best_params_


def predict(model, X_train_clean, y_train_clean, X_test, filename):
    # Standardize the training and test features
    scaler = StandardScaler()
    # scaler = RobustScaler()
    # scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_test_scaled = scaler.transform(X_test)

    # Fit the model on the cleaned and scaled training data
    model.fit(X_train_scaled, y_train_clean)

    # Predict for the test set
    y_pred = model.predict(X_test_scaled)

    # Save the predicted results to an npy file
    np.save(filename, y_pred)
    print(f"Predictions saved to {filename}")

# Split the dataset into training and validation sets


def split_data(X_train, y_train, test_size=0.2, random_state=42):
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=test_size, random_state=random_state)
    return X_train_split, X_val_split, y_train_split, y_val_split


def ridge_L2_regularization_with_validation(X_train_clean, y_train_clean, X_val, y_val):

    scaler = StandardScaler()
    # scaler = RobustScaler()
    # scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_val_scaled = scaler.transform(X_val)

    ridge = Ridge()
    param_grid = {'alpha': [0.1, 0.2, 0.25, 0.3, 0.5, 1, 10, 100],
                  'max_iter': [250, 400, 500, 600, 750, 1000, 1500, 5000, 10000]}
    ridge_cv = GridSearchCV(ridge, param_grid, cv=5)
    ridge_cv.fit(X_train_scaled, y_train_clean)

    # Best alpha value and the corresponding score
    print(f"Best alpha (Ridge): {ridge_cv.best_params_['alpha']}")
    print(f"Best cross-validated score (Ridge): {ridge_cv.best_score_}")

    # Evaluate on the validation set
    y_val_pred = ridge_cv.predict(X_val_scaled)
    print(
        f"Ridge MSE on Validation Set: {mean_squared_error(y_val, y_val_pred)}")
    print(f"Ridge R2 on Validation Set: {r2_score(y_val, y_val_pred)}")

    plot_residuals(y_val, y_val_pred, title="ridge residual plot")

    return ridge_cv.best_params_


def lasso_L1_regularization_with_validation(X_train_clean, y_train_clean, X_val, y_val):

    scaler = StandardScaler()
    # scaler = RobustScaler()
    # scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_val_scaled = scaler.transform(X_val)

    lasso = Lasso()
    param_grid = {'alpha': [0.1, 0.2, 0.25, 0.3, 0.5, 1, 10, 100],
                  'max_iter': [250, 400, 500, 600, 750, 1000, 1500, 5000, 10000]}
    lasso_cv = GridSearchCV(lasso, param_grid, cv=5)
    lasso_cv.fit(X_train_scaled, y_train_clean)

    # Best alpha value and the corresponding score
    print(f"Best alpha (Lasso): {lasso_cv.best_params_['alpha']}")
    print(f"Best cross-validated score (Lasso): {lasso_cv.best_score_}")

    # Evaluate on the validation set
    y_val_pred = lasso_cv.predict(X_val_scaled)
    print(
        f"Lasso MSE on Validation Set: {mean_squared_error(y_val, y_val_pred)}")
    print(f"Lasso R2 on Validation Set: {r2_score(y_val, y_val_pred)}")

    plot_residuals(y_val, y_val_pred, title="lasso residual plot")

    return lasso_cv.best_params_


def plot_results():
    # Plot y_train
    plt.figure(figsize=(10, 6))
    plt.hist(y_train, bins=30, alpha=0.5, label="y_train", color='green')
    plt.legend()
    plt.title("Histogram of y_train")
    plt.show()

    # Plot the predicted values,
    plt.figure(figsize=(10, 6))
    plt.hist(np.load("ridge_predictions.npy"), bins=30, alpha=0.5,
             label="Ridge Predictions", color='blue')
    plt.hist(np.load("lasso_predictions.npy"), bins=30, alpha=0.5,
             label="Lasso Predictions", color='red')
    plt.legend()
    plt.title("Histogram of Predicted Values")
    plt.show()


def plot_residuals(y_true, y_pred, title="Residuals Plot"):
    # Calculate residuals
    residuals = y_true - y_pred

    # Plot the residuals histogram
    plt.figure(figsize=(12, 5))

    # Histogram of residuals
    plt.subplot(1, 2, 1)
    sns.histplot(residuals, kde=True, bins=30, color='blue', alpha=0.6)
    plt.title("Histogram of Residuals")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")

    # Residuals vs. Predicted values plot
    plt.subplot(1, 2, 2)
    plt.scatter(y_pred, residuals, color='green', alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(title)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")

    plt.tight_layout()
    plt.show()


def ransac_regression(X_train_clean, y_train_clean):
    scaler = StandardScaler()
    # scaler = RobustScaler()
    # scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)

    ransac = RANSACRegressor(estimator=Ridge())

    param_grid = {
        # 'alpha': [0.1, 1.0, 10.0],  # Only if using Ridge as base estimator
        'min_samples': [0.5, 0.75, 0.9],
        'residual_threshold': [5.0, 10.0, 15.0],
        'max_trials': [100, 500, 1000]
    }

    ransac_cv = GridSearchCV(ransac, param_grid, cv=5)
    ransac_cv.fit(X_train_scaled, y_train_clean)

    print(f"Best parameters (RANSAC): {ransac_cv.best_params_}")
    print(f"Best cross-validated score (RANSAC): {ransac_cv.best_score_}")

    return ransac_cv.best_estimator_


def theil_sen_regression(X_train_clean, y_train_clean):

    scaler = StandardScaler()
    # scaler = RobustScaler()
    # scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)

    theil_sen = TheilSenRegressor()

    param_grid = {
        'max_subpopulation': [1e3, 1e4, 1e5],
        'n_subsamples': [None, 100, 150],
        'fit_intercept': [True, False]
    }

    theil_sen_cv = GridSearchCV(theil_sen, param_grid, cv=5)
    theil_sen_cv.fit(X_train_scaled, y_train_clean)

    print(f"Best parameters (Theil-Sen): {theil_sen_cv.best_params_}")
    print(
        f"Best cross-validated score (Theil-Sen): {theil_sen_cv.best_score_}")

    return theil_sen_cv.best_estimator_


def ransac_regression_with_validation(X_train_clean, y_train_clean, X_val, y_val):
    scaler = StandardScaler()

    ransac = RANSACRegressor(estimator=Ridge)

    param_grid = {
        # 'estimator__alpha': [0.1, 1.0, 10.0],  # Only if using Ridge as base estimator
        'min_samples': [0.5, 0.75, 0.9],
        'residual_threshold': [5.0, 10.0, 15.0],
        'max_trials': [100, 500, 1000]
    }

    ransac_cv = GridSearchCV(ransac, param_grid, cv=5)
    ransac_cv.fit(X_train_clean, y_train_clean)

    print(f"Best parameters (RANSAC): {ransac_cv.best_params_}")
    print(f"Best cross-validated score (RANSAC): {ransac_cv.best_score_}")

    y_val_pred = ransac_cv.predict(X_val)
    print(
        f"RANSAC MSE on Validation Set: {mean_squared_error(y_val, y_val_pred)}")
    print(f"RANSAC R2 on Validation Set: {r2_score(y_val, y_val_pred)}")

    plot_residuals(y_val, y_val_pred, title="RANSAC Residuals Plot")

    return ransac_cv.best_estimator_


def theil_sen_regression_with_validation(X_train_clean, y_train_clean, X_val, y_val):
    scaler = StandardScaler()
    # scaler = RobustScaler()
    # scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)

    theil_sen = TheilSenRegressor()

    param_grid = {
        'max_subpopulation': [1e3, 1e4, 1e5],
        'n_subsamples': [None, 100, 150],
        'fit_intercept': [True, False]
    }

    theil_sen_cv = GridSearchCV(theil_sen, param_grid, cv=5)
    theil_sen_cv.fit(X_train_scaled, y_train_clean)

    print(f"Best parameters (Theil-Sen): {theil_sen_cv.best_params_}")
    print(
        f"Best cross-validated score (Theil-Sen): {theil_sen_cv.best_score_}")

    y_val_pred = theil_sen_cv.predict(X_val)
    print(
        f"Theil-Sen MSE on Validation Set: {mean_squared_error(y_val, y_val_pred)}")
    print(f"Theil-Sen R2 on Validation Set: {r2_score(y_val, y_val_pred)}")

    plot_residuals(y_val, y_val_pred, title="Theil-Sen Residuals Plot")

    return theil_sen_cv.best_estimator_


if __name__ == "__main__":
    data_info(X_train, y_train)

    print("Outlier Detection with Z-score:")
    X_train_clean_z, y_train_clean_z = z_outlier_detection(X_train, y_train)
    print("Outlier Detection with local outlier factor:")
    X_train_clean_lof, y_train_clean_lof = local_outlier_factor(
        X_train, y_train)
    print("Outlier Detection with MAD:")
    X_train_clean_mad, y_train_clean_mad = outlier_detection_mad(
        X_train, y_train)
    print("Outlier Detection with IQR:")
    X_train_clean_iqr, y_train_clean_iqr = iqr_outlier_removal(
        X_train, y_train)

    # Regularization with Ridge and Lasso after outlier removal and scaling
    """
    print("Regularization with Ridge using z-score outlier removal:")
    best_params_ridge_z = ridge_L2_regularization(X_train_clean_z, y_train_clean_z)
    print("Regularization with Lasso using z-score outlier removal:")
    best_params_lasso_z = lasso_L1_regularization(X_train_clean_z, y_train_clean_z)
    print()
    print("Regularization with Ridge using local outlier factor removal:")
    best_params_ridge_lof = ridge_L2_regularization(X_train_clean_lof, y_train_clean_lof)
    print("Regularization with Lasso using local outlier factor removal:")
    best_params_lasso_lof = lasso_L1_regularization(X_train_clean_lof, y_train_clean_lof)
    print()
    print("Regularization with Ridge using mad outlier removal:")
    best_params_ridge_mad = ridge_L2_regularization(X_train_clean_mad, y_train_clean_mad)
    print("Regularization with Lasso using mad outlier removal:")
    best_params_lasso_mad = lasso_L1_regularization(X_train_clean_mad, y_train_clean_mad)
    print()
    print("Regularization with Ridge using iqr outlier removal:")
    best_params_ridge_iqr = ridge_L2_regularization(X_train_clean_iqr, y_train_clean_iqr)
    print("Regularization with Lasso using iqr outlier removal:")
    best_params_lasso_iqr = lasso_L1_regularization(X_train_clean_iqr, y_train_clean_iqr)
    """
    print("\n"*3)

    print("Outlier Detection with MAD")
    X_train_clean, y_train_clean = outlier_detection_mad(X_train, y_train)
    print("X_train_clean shape: ", X_train_clean.shape)
    print("Y_train_clean shape: ", y_train_clean.shape)
    print()

    print("Regularization with Ridge")
    best_params_ridge = ridge_L2_regularization(X_train_clean, y_train_clean)
    print("Regularization with Lasso")
    best_params_lasso = lasso_L1_regularization(X_train_clean, y_train_clean)
    print("Huber model:")
    best_params_huber = huber_regression(X_train_clean, y_train_clean)
    # print("Ransac model:")
    # best_params_ransac = ransac_regression(X_train, y_train)
    # print("Theil-Sen model:")
    # best_params_theil_sen = theil_sen_regression(X_train, y_train)

    print(
        f"\n\nBest params ridge: \n- alpha: {best_params_ridge['alpha']}\n- max_iter: {best_params_ridge['max_iter']}")
    print(
        f"\n\nBest params lasso: \n- alpha: {best_params_lasso['alpha']}\n- max_iter: {best_params_lasso['max_iter']}")
    print(
        f"\n\nBest params Huber: \n- alpha: {best_params_huber['alpha']}\n- epsilon: {best_params_huber['epsilon']}")
    # print(f"\n\nBest params Ransac: \n- min_samples: {best_params_ransac["min_samples"]}\n- residual_threshold: {best_params_ransac["residual_threshold"]}\n- max_trials: {best_params_ransac["max_trials"]}")
    # print(f"\n\nBest params Theil-Sen: \n- max_subpopulation: {best_params_theil_sen["max_subpopulation"]}\n- n_subsamples: {best_params_theil_sen["n_subsamples"]}\n- fit_intercept: {best_params_theil_sen["fit_intercept"]}")
    print("\n")

    best_ridge = Ridge(
        alpha=best_params_ridge["alpha"], max_iter=best_params_ridge["max_iter"])
    best_lasso = Lasso(
        alpha=best_params_lasso["alpha"], max_iter=best_params_lasso["max_iter"])
    best_huber = HuberRegressor(
        alpha=best_params_huber["alpha"], epsilon=best_params_huber["epsilon"])

    print("\n\nTesting with validation set: \n")

    X_train_split, X_val_split, y_train_split, y_val_split = split_data(
        X_train_clean, y_train_clean)
    best_params_lasso_val = ridge_L2_regularization_with_validation(
        X_train_split, y_train_split, X_val_split, y_val_split)
    print()
    best_params_lasso_val = lasso_L1_regularization_with_validation(
        X_train_split, y_train_split, X_val_split, y_val_split)
    print()
    best_params_huber_val = huber_regression_with_validation(
        X_train_clean, y_train_clean, X_val_split, y_val_split)
    # print()
    # best_params_theil_sen_val = theil_sen_regression_with_validation(X_train_clean, y_train_clean, X_val_split, y_val_split)
    # print()
    # best_params_ransac_val = ransac_regression_with_validation(X_train_clean, y_train_clean, X_val_split, y_val_split)

    # Predict and save the test results using Ridge and Lasso
    predict(
        best_ridge, X_train_clean, y_train_clean, X_test, "ridge_predictions.npy")
    predict(
        best_lasso, X_train_clean, y_train_clean, X_test, "lasso_predictions.npy")
    predict(
        best_huber, X_train_clean, y_train_clean, X_test,  "huber_predictions.npy")
