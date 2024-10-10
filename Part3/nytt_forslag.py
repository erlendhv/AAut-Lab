import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load the data
Xtest1 = np.load('Part3/Xtest1.npy')
Xtrain1_extra = np.load('Part3/Xtrain1_extra.npy')
Xtrain1 = np.load('Part3/Xtrain1.npy')
Ytrain1 = np.load('Part3/Ytrain1.npy')

# Split the training data
X_train, X_val, y_train, y_val = train_test_split(
    Xtrain1, Ytrain1, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
Xtest1_scaled = scaler.transform(Xtest1)
Xtrain1_extra_scaled = scaler.transform(Xtrain1_extra)

# Apply SMOTE to handle imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train_scaled, y_train)

# Reshape data for CNN
X_train_resampled_cnn = X_train_resampled.reshape(-1, 48, 48, 1)
X_val_cnn = X_val_scaled.reshape(-1, 48, 48, 1)
Xtest1_reshaped = Xtest1_scaled.reshape(-1, 48, 48, 1)
Xtrain1_extra_reshaped = Xtrain1_extra_scaled.reshape(-1, 48, 48, 1)

# CNN Model


def create_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Semi-supervised learning function


def semi_supervised_learning(model, X_train, y_train, X_unlabeled, X_val, y_val, confidence_threshold=0.95, max_iterations=5):
    for iteration in range(max_iterations):
        print(f"\nIteration {iteration + 1}")

        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        # Predict on unlabeled data
        predictions = model.predict(X_unlabeled)

        # Select confident predictions
        confident_idx = np.where((predictions > confidence_threshold) | (
            predictions < (1 - confidence_threshold)))[0]
        new_X = X_unlabeled[confident_idx]
        new_y = (predictions[confident_idx] > 0.5).astype(int).flatten()

        # Add new labeled data to training set
        X_train = np.concatenate([X_train, new_X])
        y_train = np.concatenate([y_train, new_y])

        # Remove labeled data from unlabeled set
        X_unlabeled = np.delete(X_unlabeled, confident_idx, axis=0)

        # Evaluate the model
        y_val_pred = (model.predict(X_val) > 0.5).astype(int).flatten()
        f1 = f1_score(y_val, y_val_pred)
        print(f"F1 Score: {f1:.4f}")
        print(f"Newly labeled samples: {len(new_X)}")
        print(f"Remaining unlabeled samples: {len(X_unlabeled)}")

        # Break if all unlabeled data has been labeled
        if len(X_unlabeled) == 0:
            break

    return model, X_train, y_train


# Create and train the model with semi-supervised learning
cnn_model = create_cnn_model()
cnn_model, X_train_final, y_train_final = semi_supervised_learning(
    cnn_model, X_train_resampled_cnn, y_train_resampled,
    Xtrain1_extra_reshaped, X_val_cnn, y_val
)

# Final evaluation
y_val_pred = (cnn_model.predict(X_val_cnn) > 0.5).astype(int).flatten()
final_f1 = f1_score(y_val, y_val_pred)
print("\nFinal Model Evaluation:")
print(classification_report(y_val, y_val_pred))
print(f"Final F1 Score: {final_f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Semi-supervised CNN')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Make predictions on test set
y_test_pred = (cnn_model.predict(Xtest1_reshaped) > 0.5).astype(int).flatten()

# Save predictions to file
np.save('Part3/test_predictions.npy', y_test_pred)
print("Predictions saved to 'test_predictions.npy'")
