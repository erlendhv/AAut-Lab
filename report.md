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

### Outlier Removal Methods:

#### 1. **Z-score**

The Z-score measures how many standard deviations a data point is from the mean of the dataset. It is used to identify outliers based on a threshold (e.g., 3), where any data point with a Z-score greater than the threshold is considered an outlier.

**Strengths**:

- **Simple to implement**: Just requires mean and standard deviation.
- **Effective for normal distributions**: Works well when the data follows a Gaussian distribution.

**Weaknesses**:

- **Sensitive to non-normal distributions**: It can misclassify outliers in skewed distributions.
- **Sensitive to extreme values**: A few extreme outliers can heavily distort the mean and standard deviation, which makes it harder to identify other outliers.

#### 2. **Local Outlier Factor (LOF)**

LOF is a density-based method for outlier detection. It compares the local density of a point to the local densities of its neighbors. Points that have a significantly lower density than their neighbors are considered outliers.

**Strengths**:

- **Good for complex datasets**: Handles multi-dimensional data and varying densities.
- **Effective for local outliers**: Can detect outliers that are close to dense regions but have lower densities compared to their neighbors.

**Weaknesses**:

- **Computationally expensive**: It requires calculating the density of each point and its neighbors, making it slower for large datasets.
- **Parameter sensitivity**: The results can vary significantly based on the number of neighbors (`k`) used for comparison.

#### 3. **Interquartile Range (IQR)**

IQR is based on the range between the first quartile (25th percentile) and the third quartile (75th percentile). Points falling below the first quartile or above the third quartile by more than 1.5 times the IQR are considered outliers.

**Strengths**:

- **Robust to extreme values**: Less affected by extreme outliers because it only uses the middle 50% of the data.
- **Simple to implement**: Only requires calculating the quartiles.

**Weaknesses**:

- **Insensitive to small datasets**: It may not effectively detect outliers in very small datasets or datasets with little variance.
- **Misses subtle outliers**: Can miss outliers that don’t lie far from the quartiles but are still significantly different from the general data trend.

---

### Regression Methods:

#### 1. **Huber Regression**

Huber regression is a robust regression method that behaves like linear regression for small errors and like least absolute deviation (LAD) for large errors. It reduces the effect of outliers by using a loss function that transitions from quadratic to linear beyond a certain threshold.

**Strengths**:

- **Robust to outliers**: Less sensitive to outliers than ordinary least squares (OLS), especially when the majority of the data is linear.
- **Combines benefits of OLS and LAD**: Gives a balance between robustness (from LAD) and efficiency (from OLS).

**Weaknesses**:

- **Threshold selection**: Requires a parameter to define the threshold for transitioning between the two loss functions, which can be difficult to tune.
- **Not fully resistant to extreme outliers**: If too many outliers are present, performance may degrade.

#### 2. **RANSAC (Random Sample Consensus)**

RANSAC is a robust regression technique that randomly samples subsets of the data, fits a model to the sample, and then tests how well the model fits the entire dataset. The process is repeated multiple times to find the model that fits the largest consensus of points.

**Strengths**:

- **Highly robust to outliers**: It can fit a model to the inliers while ignoring outliers, making it effective in very noisy datasets.
- **Applicable to different types of models**: Not restricted to linear regression and can be applied to various parametric models.

**Weaknesses**:

- **Computationally expensive**: Involves multiple iterations and random sampling, which can be slow for large datasets.
- **Requires parameter tuning**: The number of iterations and the acceptable error margin need to be specified.

#### 3. **Theil-Sen Estimator**

Theil-Sen is a non-parametric regression method that estimates the slope of a line by taking the median of the slopes of all pairs of points. It is robust to outliers and assumes no prior distribution for the data.

**Strengths**:

- **Robust to outliers**: Since it uses the median of slopes, it is less influenced by extreme values.
- **Distribution-free**: Does not assume any underlying distribution for the data.

**Weaknesses**:

- **Computationally expensive**: The method involves computing the slope for every pair of points, which can be inefficient for large datasets.
- **Less efficient for small noise**: While robust to large outliers, it is not as efficient as OLS when noise is small and the dataset is relatively clean.

---

These methods are useful when dealing with noisy or imbalanced datasets, where outlier detection or robust regression models can significantly improve performance.

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

# Part 3

In this image classification task, several methods are applied to achieve and evaluate performance on a binary classification problem. Here’s a breakdown of the key methods used and why they are important:

1. **Data Preprocessing**:

   - **StandardScaler**: This is used to normalize the data by scaling features to a standard normal distribution. This is crucial for machine learning models like SVM and XGBoost, which perform better when data is normalized.
   - **SMOTE (Synthetic Minority Over-sampling Technique)**: To handle class imbalance, SMOTE generates synthetic samples of the minority class. This ensures the model does not get biased toward the majority class, improving generalization for imbalanced data.

2. **Convolutional Neural Networks (CNNs)**:

   - **Modeling with CNN**: CNNs are a natural choice for image classification because they can automatically detect spatial hierarchies in images (e.g., edges, textures, shapes). This is done through convolutional layers, which learn to detect these patterns, and pooling layers, which reduce spatial dimensions while preserving features.
   - **Data Reshaping**: The images are reshaped to a 48x48 grayscale format before feeding them into the CNN. This is necessary since CNN layers expect data in a specific shape (height, width, and channels).
   - **Two CNN Architectures**: The code includes a standard CNN architecture and an alternative CNN. The alternative architecture introduces more layers and channels to detect features of different sizes and complexity, which may help improve performance in certain image tasks.

3. **Non-Deep Learning Models**:

   - **Support Vector Machine (SVM)**: The SVM with a radial basis function (RBF) kernel is used to classify images in a transformed feature space. SVMs work well for smaller, linearly separable datasets but may struggle with complex data like images. The model uses a class-weight balancing to handle class imbalance, though SVM is often less effective for image classification than CNNs.
   - **Random Forest**: A Random Forest classifier, which is an ensemble of decision trees, is included. This method can work well for tabular data and some image classification tasks but lacks the feature extraction capabilities of CNNs.
   - **XGBoost**: Another ensemble method that applies gradient boosting to decision trees. Like Random Forest, it is strong with tabular data but can be outperformed by CNNs in image tasks.

4. **Evaluation**:

   - **F1 Score**: The F1 score is used to evaluate model performance, considering both precision and recall. This is crucial for imbalanced datasets, where accuracy alone may be misleading.
   - **Confusion Matrix**: This helps visualize the classification results, showing the model’s ability to correctly classify true positives and true negatives.
   - **Learning Curves**: Plotting the learning curves for accuracy and loss helps monitor the model’s performance during training and identify issues like overfitting.

5. **Result Comparison**:
   The models are compared based on their F1 scores, with CNN achieving the best results, likely due to its ability to capture complex patterns in the image data. SVM performs the worst, as it may not handle the complexity of image data well without feature engineering.

In summary, the methods range from traditional machine learning (SVM, Random Forest, XGBoost) to deep learning (CNN), with CNN performing best because of its ability to automatically learn spatial hierarchies and patterns in image data.

### XGBoost (Extreme Gradient Boosting)

**XGBoost** is an efficient and scalable implementation of gradient-boosted decision trees. It improves traditional boosting algorithms by incorporating regularization, handling missing values, and optimizing for both speed and accuracy. XGBoost is widely used in machine learning competitions due to its performance and flexibility.

### Strengths of XGBoost:

1. **Highly Accurate**:

   - XGBoost generally provides state-of-the-art performance for both classification and regression tasks. It often outperforms traditional machine learning models due to its boosting mechanism, which iteratively improves upon previous models.

2. **Handles Missing Data**:

   - It can automatically learn how to handle missing data during training, which simplifies preprocessing.

3. **Regularization**:

   - XGBoost includes both L1 (Lasso) and L2 (Ridge) regularization, which helps prevent overfitting and makes the model more robust, especially when dealing with high-dimensional data or complex relationships.

4. **Efficient Computation**:

   - It is highly optimized for speed, using techniques like:
     - **Parallelization**: Boosting is typically sequential, but XGBoost optimizes it to run faster by parallelizing computations during tree construction.
     - **Tree Pruning**: XGBoost uses a process called “max_depth pruning” to ensure that trees don’t grow too complex.
     - **Block structure**: XGBoost uses an internal block structure to improve cache efficiency during training.

5. **Feature Importance**:

   - XGBoost provides insights into which features are most important for prediction, making it easier to interpret results and understand which variables have the most impact.

6. **Customizable**:

   - Users can control aspects like tree depth, learning rate, and the number of trees, which offers flexibility to tune the model for specific problems.

7. **Handles Imbalanced Datasets**:

   - XGBoost has options for handling imbalanced datasets by adjusting the `scale_pos_weight` parameter for classification tasks or using evaluation metrics suited for imbalanced data like AUC (Area Under the Curve).

8. **Cross-validation and Early Stopping**:
   - It can incorporate cross-validation during training and apply early stopping when the model's performance stops improving, which helps avoid overfitting.

### Weaknesses of XGBoost:

1. **Complex Hyperparameter Tuning**:

   - XGBoost has many hyperparameters (learning rate, maximum depth, regularization terms, etc.), which makes tuning more complex and time-consuming compared to simpler models. Improper tuning can lead to overfitting or underfitting.

2. **Computationally Expensive for Large Datasets**:

   - While XGBoost is optimized for speed, it can still be computationally intensive for extremely large datasets, especially with deep trees. Memory usage can become an issue for high-dimensional datasets.

3. **Sensitive to Noisy Data**:

   - Like other tree-based models, XGBoost can overfit noisy datasets if not properly tuned, especially when trees become overly complex. Regularization helps, but it needs to be correctly applied.

4. **Less Interpretable than Simple Models**:

   - Despite offering feature importance scores, XGBoost is less interpretable than simpler models like linear regression or decision trees. Understanding the influence of each individual decision tree in the ensemble can be difficult.

5. **Requires Extensive Preprocessing**:

   - While XGBoost handles missing data well, it still requires other preprocessing steps such as encoding categorical variables (since it only handles numerical data) and scaling, especially when using regularization.

6. **Potential for Overfitting**:
   - If too many boosting rounds are used without early stopping or proper regularization, XGBoost can overfit the training data, leading to poor generalization on unseen data.

### Summary of XGBoost:

- **Strengths**: High accuracy, efficient computation, automatic handling of missing data, built-in regularization, feature importance, works well on imbalanced datasets, customizable, and supports early stopping.
- **Weaknesses**: Complex tuning, computationally intensive for very large datasets, sensitive to noise, less interpretable, requires preprocessing, and can overfit without proper regularization.

XGBoost is particularly effective for structured/tabular data, especially when the relationships between features are complex and non-linear. However, the model's complexity and resource requirements make it most suitable for problems where performance is critical and the computational cost can be justified.
