# Part 1

In this problem, the goal was to estimate the relationship between toxic algae concentration and several independent variables using linear regression, while accounting for noisy and outlier-prone data. You tested various methodologies to identify the best model, ultimately choosing MAD outlier detection, Ridge regression with L2 regularization, and scaling with StandardScaler. Here's a breakdown of the methods and why Ridge worked best:

### 1. **Outlier Detection**

- **Z-score vs. MAD**:
  - You initially explored Z-score-based outlier detection but found that it detected only a small number of outliers. Z-scores can struggle when data contains heavy-tailed noise (as in the human error affecting the dependent variable).
  - **MAD (Median Absolute Deviation)**: MAD, which measures dispersion based on the median rather than the mean, is more robust to outliers. In datasets like yours, where 25% of samples are affected by significant human error, MAD is less sensitive to extreme values, making it a better choice for detecting outliers in the presence of large deviations.

### 2. **Regularization Techniques**

- **Ridge (L2) Regularization**:
  - Ridge regression adds a penalty on the size of the coefficients to prevent overfitting. Since the dataset contains instrument noise and human error, overfitting would likely occur if we rely solely on the ordinary least squares (OLS) method. Ridge regression reduces the model complexity by shrinking the coefficients, making it more resilient to the noise in your data. This helps avoid fitting to the noisy components (like ξ in the equation) and focuses on the underlying signal.
  - **Why Lasso wasn’t as effective**: Lasso (L1) regularization can drive some coefficients to zero, which works well when many variables are irrelevant. However, in your case, all five independent variables (temperature, wind speed, etc.) are likely important, so shrinking any of them to zero might lead to underfitting. Ridge, which distributes the penalty evenly, keeps all coefficients in the model while controlling their magnitude, making it more suitable here.

### 3. **Scaling**

- **StandardScaler**:
  - Since Ridge regularization is sensitive to the scale of features, standardizing the data (using StandardScaler) was essential. It ensured that all independent variables contributed equally to the model, rather than being dominated by those with larger scales (e.g., illumination vs. wind direction).

### 4. **Why Ridge Performed Best**

- **Noise and Regularization**: The presence of Gaussian noise and human error in your data made regularization crucial. Ridge’s ability to handle multicollinearity (high correlation between variables) and prevent overfitting to noisy data helped improve model performance.
- **All Features are Important**: Since each independent variable likely contributes to toxic algae concentration, Ridge’s ability to shrink but not eliminate coefficients (unlike Lasso) preserved the full model structure while controlling complexity.

This combination of MAD for robust outlier detection and Ridge with standard scaling was effective because it managed noise and outliers while preventing overfitting to the erratic elements of the dataset.

# Part 2
