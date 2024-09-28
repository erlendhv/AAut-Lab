from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures

# Not relevant, try to find linear relationship between input and output.
# Polynomial features implies a non-linear relationship.

def ridge_L2_with_polynomial(X_train_clean, y_train_clean, degree=2):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train_scaled)
    ridge = Ridge(max_iter=20000)
    param_grid = {'alpha': [1, 2, 4, 6, 8, 10, 30, 70, 100, 130, 160, 200], 'max_iter': [1000, 5000, 10000]}
    ridge_cv = GridSearchCV(ridge, param_grid, cv=5)
    ridge_cv.fit(X_train_poly, y_train_clean)
    print(f"Best alpha: {ridge_cv.best_params_['alpha']}")
    print(f"Best cross-validated score: {ridge_cv.best_score_}")
    return ridge_cv.best_params_


# Not relevant, try to find linear relationship between input and output.
# Polynomial features implies a non-linear relationship.
def lasso_L1_with_polynomial(X_train_clean, y_train_clean, degree=2):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train_scaled)
    lasso = Lasso(max_iter=20000)
    param_grid = {'alpha': [0.01, 0.1, 1, 3, 5, 7], 'max_iter': [1000, 5000, 10000]}
    lasso_cv = GridSearchCV(lasso, param_grid, cv=5)
    lasso_cv.fit(X_train_poly, y_train_clean)
    print(f"Best alpha: {lasso_cv.best_params_['alpha']}")
    print(f"Best cross-validated score: {lasso_cv.best_score_}")
    return lasso_cv.best_params_
