import numpy as np
import matplotlib.pyplot as plt

Xtest1 = np.load('Part3/Xtest1.npy')
Xtrain1_extra = np.load('Part3/Xtrain1_extra.npy')
Xtrain1 = np.load('Part3/Xtrain1.npy')
Ytrain1 = np.load('Part3/Ytrain1.npy')


print('Xtest1:', Xtest1.shape)
print('Xtrain1_extra:', Xtrain1_extra.shape)
print('Xtrain1:', Xtrain1.shape)
print('Ytrain1:', Ytrain1.shape)

# Plot the training data, connecting the points with a line
# plt.plot(range(len(Xtrain1)), Xtrain1, 'r-', label='Xtrain1')
# plt.plot(range(len(Ytrain1)), Ytrain1, 'b-', label='Ytrain1')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.title('Training Data')
# plt.legend()
# plt.show()
