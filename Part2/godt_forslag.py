import numpy as np
from scipy import linalg


def normalize_data(data):
    return (data - np.mean(data)) / np.std(data)


def denormalize_data(normalized_data, original_mean, original_std):
    return normalized_data * original_std + original_mean


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

        start = max(0, i-d-m)
        end = i-d+1
        if start < end:
            u_slice = u[start:end][::-1]
            phi[n:n+len(u_slice)] = u_slice

        y_pred[i] = np.dot(phi, theta)

    return y_pred


def evaluate_model(y_true, y_pred):
    residuals = y_true - y_pred
    # Use np.abs and np.log1p for improved numerical stability
    return np.sum(np.log1p(np.abs(residuals)))


# Load and normalize data
u_train = np.load('Part2/u_train.npy')
y_train = np.load('Part2/output_train.npy')
u_test = np.load('Part2/u_test.npy')

u_train_mean, u_train_std = np.mean(u_train), np.std(u_train)
y_train_mean, y_train_std = np.mean(y_train), np.std(y_train)

u_train_norm = normalize_data(u_train)
y_train_norm = normalize_data(y_train)
u_test_norm = (u_test - u_train_mean) / u_train_std

# Grid search for best parameters
best_sse = float('inf')
best_params = None
best_theta = None

for n in range(1, 10):
    for m in range(10):
        for d in range(10):
            theta = train_arx_model(y_train_norm, u_train_norm, n, m, d)
            y_pred_norm = predict_arx(
                u_train_norm, y_train_norm[:n], theta, n, m, d)
            sse = evaluate_model(y_train_norm[n:], y_pred_norm[n:])

            if sse < best_sse:
                best_sse = sse
                best_params = (n, m, d)
                best_theta = theta

print(
    f"Best parameters: n={best_params[0]}, m={best_params[1]}, d={best_params[2]}")
print(f"Best log-sum-abs-residuals: {best_sse}")

# Predict on test set
y_test_pred_norm = predict_arx(
    u_test_norm, y_train_norm[-best_params[0]:], best_theta, *best_params)

# Denormalize predictions
y_test_pred = denormalize_data(y_test_pred_norm, y_train_mean, y_train_std)

# Save last 400 samples
np.save('Part2/y_test_pred_last_400.npy', y_test_pred[-400:])

print("Prediction complete. Last 400 samples saved to 'y_test_pred_last_400.npy'.")
