import numpy as np


u_train = np.load('Part2/u_train.npy')
output_train = np.load('Part2/output_train.npy')
u_test = np.load('Part2/u_test.npy')


# Plot the training data, connecting the points with a line
import matplotlib.pyplot as plt
plt.plot(range(len(u_train)), u_train, 'r-', label='u_train')
plt.plot(range(len(output_train)), output_train, 'b-', label='output_train')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Training Data')
plt.legend()
# plt.show()

positive5 = []
negative5 = []
outliers = []
positives = []
negatives = []
zeros = []
non_matches = []
for i in range(len(u_train)):
    if u_train[i] == 5:
        positive5.append(i)
    elif u_train[i] == -5:
        negative5.append(i)
    else:
        outliers.append(i)
    if output_train[i] > 0:
        positives.append(i)
    elif output_train[i] < 0:
        negatives.append(i)
    else:
        zeros.append(i)

    # Check if positive u_train is positive output
    if u_train[i] > 0 and output_train[i] < 0:
        non_matches.append(i)
    # Check if negative u_train is negative output
    elif u_train[i] < 0 and output_train[i] > 0:
        non_matches.append(i)

print('Positive 5:', len(positive5))
print('Negative 5:', len(negative5))
print('Outliers:', len(outliers))

print('Positives:', len(positives))
print('Negatives:', len(negatives))
print('Zeros:', len(zeros))

print('Non-matches:', len(non_matches))