import numpy as np
from scipy import linalg


def normalize_data(data):
    return (data - np.mean(data)) / np.std(data)


def denormalize_data(normalized_data, original_mean, original_std):
    return normalized_data * original_std + original_mean


def create_regressor_matrix(output_data, input_data, n, m, d):
    total_samples = len(output_data)
    p = max(n, d + m)
    num_features = n + m + 1
    num_rows = total_samples - p

    regressor_matrix = np.zeros((num_rows, num_features))
    target_vector = output_data[p:]

    for row in range(num_rows):
        output_lags = -output_data[row + p - n:row + p][::-1]
        input_lags = input_data[row + p -
                                d - m:row + p - d + 1][::-1]
        regressor_matrix[row, :n] = output_lags
        regressor_matrix[row, n:] = input_lags

    return regressor_matrix, target_vector


def train_arx_model(output_data, input_data, n, m, d):
    regressor_matrix, target_vector = create_regressor_matrix(
        output_data, input_data, n, m, d)
    theta, _, _, _ = linalg.lstsq(regressor_matrix, target_vector)
    return theta


def predict_arx(input_data, initial_output, theta, n, m, d):
    total_samples = len(input_data)
    predictions = np.zeros(total_samples)
    predictions[:n] = initial_output

    for i in range(n, total_samples):
        feature_vector = np.zeros(n + m + 1)
        output_lags = -predictions[i-n:i][::-1]
        feature_vector[:n] = output_lags

        input_start = max(0, i - d - m)
        input_end = i - d + 1
        if input_start < input_end:
            input_lags = input_data[input_start:input_end][::-1]
            feature_vector[n:n+len(input_lags)] = input_lags

        predictions[i] = np.dot(feature_vector, theta)

    return predictions


def evaluate_model(true_values, predicted_values):
    residuals = true_values - predicted_values
    return np.sum(residuals ** 2)  # Sum of Squared Errors (SSE)


# Load and normalize data
input_train = np.load('Part2/u_train.npy')
output_train = np.load('Part2/output_train.npy')
input_test = np.load('Part2/u_test.npy')

input_train_mean, input_train_std = np.mean(input_train), np.std(input_train)
output_train_mean, output_train_std = np.mean(
    output_train), np.std(output_train)

input_train_norm = normalize_data(input_train)
output_train_norm = normalize_data(output_train)
input_test_norm = (input_test - input_train_mean) / input_train_std

# Split the training data into training and validation sets
split_index = int(0.8 * len(output_train_norm))
input_train_split = input_train_norm[:split_index]
output_train_split = output_train_norm[:split_index]
input_val_split = input_train_norm[split_index:]
output_val_split = output_train_norm[split_index:]

# Grid search for best parameters using the validation set
best_sse = float('inf')
best_params = None
best_theta = None

for n in range(1, 10):
    for m in range(10):
        for d in range(10):
            # Train ARX model on training split
            theta = train_arx_model(
                output_train_split, input_train_split, n, m, d)
            # Predict on the validation set
            predictions_val_norm = predict_arx(
                input_val_split, output_train_split[:n], theta, n, m, d)
            sse_val = evaluate_model(output_val_split[n:], predictions_val_norm[n:])

            if sse_val < best_sse:
                best_sse = sse_val
                best_params = (n, m, d)
                best_theta = theta

print(
    f"Best parameters: n={best_params[0]}, m={best_params[1]}, d={best_params[2]}")
print(f"Best validation SSE: {best_sse}")

# Now, predict on the test set using the best parameters
test_predictions_norm = predict_arx(
    input_test_norm, output_train_norm[-best_params[0]:], best_theta, *best_params)

# Denormalize predictions
test_predictions = denormalize_data(
    test_predictions_norm, output_train_mean, output_train_std)

# Save last 400 samples
np.save('Part2/y_test_pred_last_400.npy', test_predictions[-400:])

print("Prediction complete. Last 400 samples saved to 'y_test_pred_last_400.npy'.")
