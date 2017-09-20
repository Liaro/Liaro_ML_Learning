import numpy as np
import os
import sys

import cv2
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import  DecisionTreeClassifier

from make_dataset import load_data
from visualize import plot_confusion_matrix


# NNを利用しないモデルの候補
models = {
    "knn": KNeighborsClassifier(n_jobs=2), # 時間かかる
    "dt": DecisionTreeClassifier(), # 時間かかる
    "rf": RandomForestClassifier(n_jobs=-1), # すぐ終わる
    "svm": SVC() # 時間かかる
}


if __name__ == "__main__":

    sys.path.append(os.pardir)

    if len(sys.argv) == 2 and sys.argv[1] in models.keys(): # コマンドライン引数が条件を満たしているとき
        model_name = sys.argv[1]

        # データの準備
        koma = load_data() # 駒の種類．混同行列に利用．
        class_names = koma.target_names
        x = koma.data.reshape(koma.data.shape[0], -1) # 一次元化
        y = koma.target
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        model = models[model_name] # コマンドライン引数からモデルを選択

        # 学習済みモデルがあれば利用し，なければ学習させる
        try:
            clf = joblib.load("../result/{}.pkl".format(model_name))
        except:
            clf = model.fit(x_train, y_train)

        y_pred = clf.predict(x_test)  # 予測

        # 結果の表示
        print(model.__class__.__name__)
        print("train:", clf.score(x_train, y_train))
        print("test:", clf.score(x_test, y_test))
        print("F1: ", f1_score(y_test[:len(y_pred)], y_pred, average='macro'))

        ## 正規化前の混合行列の可視化
        plot_confusion_matrix(y_test, y_pred, classes=class_names, title='Confusion matrix, without normalization')

        ##  正規化後の混合行列の可視化
        plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True, title='Normalized confusion matrix')

        plt.show()

        # モデルの保存
        joblib.dump(clf, "../result/{}.pkl".format(sys.argv[1]))

    else: # 例外処理
        print("please specify the model (knn, dt, rf or svm) like $ python non_nn.py rf ")
