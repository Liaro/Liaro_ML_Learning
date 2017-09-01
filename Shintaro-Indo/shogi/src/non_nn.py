import sys,os
sys.path.append(os.pardir)
from dataset.data import fetch_data

import numpy as np
import itertools
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.tree import  DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# グレースケール化 → 閾値処理
def binarization(x_data):
    # グレースケール化
    x_data =  np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in x_data])

    # 閾値処理
    x_data = np.array([cv2.threshold(img, 170, 255, cv2.THRESH_BINARY)[1] for img in x_data])

    return x_data


# 混同行列を描画する関数
def plot_confusion_matrix(y_test, y_pred, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    #  混同行列の作成
    cm = confusion_matrix(y_test, y_pred)

    # 正規化
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # 行の和を列ベクトル化

    plt.figure(figsize = (6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# モデル
models = [
    KNeighborsClassifier(n_jobs=2),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_jobs=-1),
    SVC()
]


if __name__ == "__main__":

    # データの読み込み
    koma = fetch_data()
    x = koma.data.reshape(koma.data.shape[0], -1) # 一次元化
    y = koma.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # 駒の種類
    class_names = koma.target_names


    model = models[2]
    clf = model.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    print(model.__class__.__name__)
    print("train:", clf.score(x_train, y_train))
    print("test:", clf.score(x_test, y_test))
    print("F1: ", f1_score(y_test[:len(y_pred)], y_pred, average='macro'))


    # 正規化前の混合行列の可視化
    plot_confusion_matrix(y_test, y_pred, classes=class_names, title='Confusion matrix, without normalization')

    # 正規化後の混合行列の可視化
    plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True, title='Normalized confusion matrix')

    plt.show()
