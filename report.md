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

### Summary of Your Solution

You approached the problem of estimating the parameters of an ARX (AutoRegressive with eXogenous input) model for a given system using **Recursive Least Squares (RLS)**, aiming to minimize the Sum of Squared Errors (SSE) on the last 400 samples of the test set. Here's an overview of your workflow:

1. **Data Normalization**: 
   - The input (`u_train`) and output (`y_train`) data were normalized before training, ensuring that the model doesn’t get biased by different scales of the data.
   
2. **Model Construction**:
   - You formulated the ARX model according to the general structure:
     \[
     y(k) + a_1y(k-1) + \cdots + a_ny(k-n) = b_0u(k-d) + \cdots + b_mu(k-d-m) + e(k)
     \]
   - This equation was reformulated as a linear regression problem \( Y = X\theta \), where \( \theta \) contained the coefficients to be estimated.
   
3. **Parameter Search**:
   - You used **grid search** over the parameters `n`, `m`, and `d`, each ranging from 1–10, 0–10, and 0–10 respectively.
   - The model was trained using RLS on the first 80% of the training data, and validated on the remaining 20%.
   
4. **Model Training**:
   - The Recursive Least Squares (RLS) method was applied iteratively, adjusting the weights (coefficients) based on the data.
   
5. **Model Evaluation**:
   - The performance of each set of parameters was evaluated using the **Sum of Squared Errors (SSE)** on the validation set, and the best combination of `n`, `m`, and `d` was chosen based on the lowest SSE.
   
6. **Test Set Prediction**:
   - After determining the best model, predictions were made on the test input (`u_test`) and the output was denormalized to produce the final results, saving the last 400 predicted samples.
   
7. **Result**:
   - The final SSE was around 9, which indicates that there is room for improvement.

### Reflection on Your Solution

While your approach was methodical, a few areas might have contributed to the relatively high SSE of 9. Here are some reflections:

1. **Data Normalization**:
   - You properly normalized the data, which is crucial for ensuring that the RLS algorithm treats all input features equally. This was done correctly and helped mitigate issues of differing scales between `u` and `y`.

2. **Grid Search and Model Selection**:
   - The grid search over `n`, `m`, and `d` parameters was a good approach. However, the ranges (0–10 for `n`, `m`, and `d`) might still have left out potentially better values. Expanding the grid search or using a more efficient search method like **random search** or **Bayesian optimization** could be more effective.

3. **Recursive Least Squares (RLS)**:
   - RLS is an efficient method for online parameter estimation, but it may sometimes suffer from issues related to numerical instability, especially if the data is noisy or ill-conditioned. Choosing the regularization factor (`lambda_`) or initializing the covariance matrix differently (e.g., higher values for `P`) might stabilize the algorithm.
   - Additionally, RLS assumes a fixed model structure. It might not adapt well to systems with changing dynamics or higher-order effects, so other algorithms such as **Extended RLS** or **Kalman filters** might be worth exploring.

4. **Overfitting Risk**:
   - Although you used a validation set, the chosen parameter set might have overfitted to this validation data, leading to a suboptimal performance on the test set. Techniques like **cross-validation** or **regularization** (to penalize overfitting) could mitigate this risk.

5. **Test Prediction Stability**:
   - Linear ARX models, especially in the context of RLS, might predict unstable results if the model's poles are not constrained. Stability constraints (e.g., ensuring that the autoregressive part has all roots inside the unit circle) could prevent the model from becoming unstable, which may have led to some of the high SSE.

### Suggestions for Improvement

1. **Extended Model Search**:
   - Expand the grid search range for parameters `n`, `m`, and `d` beyond 10. You might find better parameter values in higher ranges.
   - Try **non-uniform search techniques** like Bayesian optimization, which might find better parameters more efficiently than a brute-force grid search.

2. **Regularization**:
   - Introduce a **regularization term** in the cost function (e.g., L2 regularization) to prevent overfitting and enforce smoother coefficient estimates.
   - Alternatively, penalizing large values in the coefficient vector `theta` could stabilize the system.

3. **Cross-Validation**:
   - Use **k-fold cross-validation** on the training data instead of a fixed 80-20 split. This ensures that the model is trained and validated on different subsets, leading to more generalizable results.

4. **Stability Constraints**:
   - Impose **stability constraints** on the autoregressive part of the model, such as ensuring that the poles of the system remain inside the unit circle. This could prevent the model from producing predictions that diverge or become unstable over time.

5. **Alternative Algorithms**:
   - Consider alternative algorithms to RLS, such as:
     - **Kalman filters**: which might be more robust for time-varying systems.
     - **Subspace identification**: which is used in more advanced system identification and might yield more stable results.
   
6. **Noise Handling**:
   - Add a noise model to the ARX structure, converting it into an **ARMAX model** (which accounts for noise in a more structured way). This can help mitigate the impact of unmodeled disturbances and improve generalization.

7. **Model Simplification**:
   - Experiment with **simpler models** by reducing the number of parameters (smaller `n`, `m`, and `d`). Sometimes, a simpler model can generalize better and avoid overfitting.

By incorporating some of these suggestions, you may be able to achieve a lower SSE on the test set and improve the overall accuracy of your predictions.

### From professor

Those that performed well on 2nd regression problem:

- Were carefule with validation set: tested with the same iterative approach as with when building the model on the training set.
- Appended the test set to the training set when predicting
- Some delivered the wrong 400 values (Petter og Ole)