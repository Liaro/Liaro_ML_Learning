import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_test, y_pred, classes, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues):
    """
    混同行列を描画する関数
    """
    confusion_matrix_ = confusion_matrix(y_test, y_pred) # 混同行列

    if normalize:
        # 各行の和を列ベクトル化
        confusion_matrix_ = (confusion_matrix_.astype('float')
                            / confusion_matrix_.sum(axis=1)[:, np.newaxis])

    plt.figure(figsize = (6, 6))
    plt.imshow(confusion_matrix_, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = confusion_matrix_.max() / 2.
    for i, j in itertools.product(range(confusion_matrix_.shape[0]),
                                  range(confusion_matrix_.shape[1])):
        plt.text(j, i, format(confusion_matrix_[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confusion_matrix_[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
