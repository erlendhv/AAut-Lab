import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

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

# Function to plot sample images


def plot_sample_images(X, y, num_samples=5):
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        idx = np.random.randint(0, X.shape[0])
        axes[i].imshow(X[idx].reshape(48, 48), cmap='gray')
        axes[i].set_title(f"Label: {y[idx]}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()


# Plot class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x=Ytrain1)
plt.title("Class Distribution in Training Data")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# Plot sample images
plot_sample_images(Xtrain1, Ytrain1)

# Print class distribution
print("Class distribution:")
print(np.bincount(Ytrain1))

# Apply SMOTE to handle imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train_scaled, y_train)

print("Shape of training data after SMOTE:", X_train_resampled.shape)
print("Class distribution after SMOTE:")
print(np.bincount(y_train_resampled))

# Reshape resampled data for CNN
X_train_resampled_cnn = X_train_resampled.reshape(-1, 48, 48, 1)
X_val_cnn = X_val_scaled.reshape(-1, 48, 48, 1)
Xtest1_reshaped = Xtest1_scaled.reshape(-1, 48, 48, 1)


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

# Train and evaluate models


def train_and_evaluate(model, X_train, y_train, X_val, y_val, model_name):
    if model_name == 'CNN':
        history = model.fit(X_train.reshape(-1, 48, 48, 1), y_train,
                            validation_data=(
                                X_val.reshape(-1, 48, 48, 1), y_val),
                            epochs=10, batch_size=32, verbose=0)
        y_pred = (model.predict(X_val.reshape(-1, 48, 48, 1))
                  > 0.5).astype(int).flatten()
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

    f1 = f1_score(y_val, y_pred)
    print(f"\n{model_name} Results:")
    print(classification_report(y_val, y_pred))
    print(f"F1 Score: {f1:.4f}")

    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    return f1


# Create and train models
cnn_model = create_cnn_model()
svm_model = SVC(kernel='rbf', class_weight='balanced')
rf_model = RandomForestClassifier(
    n_estimators=100, class_weight='balanced', random_state=42)
xgb_model = XGBClassifier(use_label_encoder=False,
                          eval_metric='logloss', scale_pos_weight=1)

models = [
    (cnn_model, X_train_resampled_cnn, 'CNN'),
    (svm_model, X_train_resampled, 'SVM'),
    (rf_model, X_train_resampled, 'Random Forest'),
    (xgb_model, X_train_resampled, 'XGBoost')
]

results = {}

for model, X_train, name in models:
    f1 = train_and_evaluate(
        model, X_train, y_train_resampled, X_val_scaled, y_val, name)
    results[name] = f1

# Compare results
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values())
plt.title('F1 Scores Comparison')
plt.xlabel('Models')
plt.ylabel('F1 Score')
plt.ylim(0, 1)
for i, v in enumerate(results.values()):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
plt.show()

# Select best model
best_model_name = max(results, key=results.get)
print(f"\nBest performing model: {best_model_name}")

# Make predictions on test set using the best model
best_model = [model for model, _, name in models if name == best_model_name][0]
if best_model_name == 'CNN':
    y_test_pred = (best_model.predict(Xtest1_reshaped)
                   > 0.5).astype(int).flatten()
else:
    y_test_pred = best_model.predict(Xtest1_scaled)

print("\nPredictions on test set:")
print(y_test_pred)

# Save predictions to file
np.save('test_predictions.npy', y_test_pred)
print("Predictions saved to 'test_predictions.npy'")
