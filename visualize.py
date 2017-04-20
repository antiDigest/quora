from sklearn.metrics import log_loss, confusion_matrix
import pandas as pd
import numpy as np

import itertools
import numpy as np
import matplotlib.pyplot as plt

x_train = pd.read_csv('data/train.csv')
y_train = x_train['is_duplicate']
ids = x_train['id']

out = pd.read_csv('data/submit_train.csv')
y_pred = out['is_duplicate']


def output(x):
    if x >= 0.075:
        return 1
    else:
        return 0


def plot_confusion_matrix(cm, classes, img, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig('visual/' + img)


print log_loss(y_train, y_pred)

y = []
for row in y_pred.tolist():
    y.append(output(row))

y = np.array(y)

cnf_matrix = confusion_matrix(y_train, y)
np.set_printoptions(precision=2)

class_names = ['Duplicate', 'Not Duplicate']

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, img='doc2vec.png',
                      title='Confusion matrix, without normalization')

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, img='doc2vec_normalised.png', normalize=True,
                      title='Confusion matrix, normalised')
