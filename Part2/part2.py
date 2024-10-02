import matplotlib.pyplot as plt
import numpy as np


u_train = np.load('Part2/u_train.npy')
output_train = np.load('Part2/output_train.npy')
u_test = np.load('Part2/u_test.npy')


# Plot the training data, connecting the points with a line
plt.plot(range(len(u_train)), u_train, 'r-', label='u_train')
plt.plot(range(len(output_train)), output_train, 'b-', label='output_train')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Training Data')
plt.legend()
plt.show()

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

print('Positive 5 train:', len(positive5))
print('Negative 5 train:', len(negative5))
print('Outliers train:', len(outliers))

print('Positives train:', len(positives))
print('Negatives train:', len(negatives))
print('Zeros train:', len(zeros))

print('Non-matches train:', len(non_matches))

y_pred = np.load('y_test_pred_last_400.npy')
u_test = u_test[-400:]
print('y_pred:', y_pred.shape)
# plt.plot(range(len(output_train)), output_train, 'b-', label='output_train')
plt.plot(range(len(y_pred)), y_pred, 'r-', label='y_test_pred')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Training Data and Predictions')
plt.legend()
plt.show()

print("u_test: ", u_test.shape)

positive5 = []
negative5 = []
outliers = []
positives = []
negatives = []
zeros = []
non_matches = []
for i in range(len(u_test)):
    if u_test[i] == 5:
        positive5.append(i)
    elif u_test[i] == -5:
        negative5.append(i)
    else:
        outliers.append(i)
    if y_pred[i] > 0:
        positives.append(i)
    elif y_pred[i] < 0:
        negatives.append(i)
    else:
        zeros.append(i)

    # Check if positive u_train is positive output
    if u_test[i] > 0 and y_pred[i] < 0:
        non_matches.append(i)
    # Check if negative u_train is negative output
    elif u_test[i] < 0 and y_pred[i] > 0:
        non_matches.append(i)

print('Positive 5 test:', len(positive5))
print('Negative 5 test:', len(negative5))
print('Outliers test:', len(outliers))

print('Positives test:', len(positives))
print('Negatives test:', len(negatives))
print('Zeros test:', len(zeros))

print('Non-matches test:', len(non_matches))
