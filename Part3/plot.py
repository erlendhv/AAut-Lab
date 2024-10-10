import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns


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
