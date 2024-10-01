import numpy as np
from scipy import linalg

def create_regressor_matrix(y, u, n, m, d):
    N = len(y)
    p = max(n, d + m)
    X = np.zeros((N - p, n + m + 1))
    
    for i in range(p, N):
        X[i-p, :n] = -y[i-n:i][::-1]
        X[i-p, n:] = u[i-d-m:i-d+1][::-1]
    
    return X, y[p:]

def train_arx_model(y, u, n, m, d):
    X, Y = create_regressor_matrix(y, u, n, m, d)
    theta = linalg.lstsq(X, Y)[0]
    return theta

def predict_arx(u, y_init, theta, n, m, d):
    N = len(u)
    y_pred = np.zeros(N)
    y_pred[:n] = y_init
    
    for i in range(n, N):
        phi = np.zeros(n + m + 1)
        phi[:n] = -y_pred[i-n:i][::-1]
        phi[n:] = u[i-d-m:i-d+1][::-1]
        y_pred[i] = np.dot(phi, theta)
    
    return y_pred

def evaluate_model(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)

# Load data
u_train = np.load('Part2/u_train.npy')
y_train = np.load('Part2/output_train.npy')
u_test = np.load('Part2/u_test.npy')

# Grid search for best parameters
best_sse = float('inf')
best_params = None
best_theta = None

for n in range(1, 10):
    for m in range(10):
        for d in range(10):
            theta = train_arx_model(y_train, u_train, n, m, d)
            y_pred = predict_arx(u_train, y_train[:n], theta, n, m, d)
            sse = evaluate_model(y_train[n:], y_pred[n:])
            
            if sse < best_sse:
                best_sse = sse
                best_params = (n, m, d)
                best_theta = theta

print(f"Best parameters: n={best_params[0]}, m={best_params[1]}, d={best_params[2]}")

# Predict on test set
y_test_pred = predict_arx(u_test, y_train[-best_params[0]:], best_theta, *best_params)

# Save last 400 samples
np.save('y_test_pred_last_400.npy', y_test_pred[-400:])

print("Prediction complete. Last 400 samples saved to 'y_test_pred_last_400.npy'.")