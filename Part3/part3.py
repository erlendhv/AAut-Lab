import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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

def create_alternative_cnn_model():
    model = Sequential([
        # First block: detecting small features, few channels
        Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(48, 48, 1)),
        MaxPooling2D((2, 2), strides=2),
        Dropout(0.25),

        # Second block: detecting medium features, more channels
        Conv2D(32, (5, 5), activation='relu', padding='same'),
        MaxPooling2D((2, 2), strides=2),
        Dropout(0.25),

        # Third block: detecting larger features, even more channels
        Conv2D(64, (7, 7), activation='relu', padding='same'),
        MaxPooling2D((2, 2), strides=2),
        Dropout(0.5),

        # Flatten and dense layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Sigmoid for binary classification
    ])
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def plot_learning_curve(history, model_name):
    """Plot learning curve based on training history"""
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{model_name} - Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{model_name} - Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.show()

# Train and evaluate models
def train_and_evaluate(model, X_train, y_train, X_val, y_val, model_name):
    if model_name == 'CNN' or model_name == 'Alternative CNN':
        history = model.fit(X_train.reshape(-1, 48, 48, 1), y_train,
                            validation_data=(
                                X_val.reshape(-1, 48, 48, 1), y_val),
                            epochs=10, batch_size=32, verbose=0)
        plot_learning_curve(history, model_name)
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
alt_cnn_model = create_alternative_cnn_model()

models = [
    (cnn_model, X_train_resampled_cnn, 'CNN'),
    (svm_model, X_train_resampled, 'SVM'),
    (rf_model, X_train_resampled, 'Random Forest'),
    (xgb_model, X_train_resampled, 'XGBoost'),
    (alt_cnn_model, X_train_resampled_cnn, 'Alternative CNN')
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
np.save('Part3/test_predictions.npy', y_test_pred)
print("Predictions saved to 'test_predictions.npy'")
