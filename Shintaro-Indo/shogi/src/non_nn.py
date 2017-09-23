import argparse
import sys
import os

import numpy as np
import cv2
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import  DecisionTreeClassifier

from make_dataset import LoadData
from visualize import plot_confusion_matrix


# NNを利用しないモデルの候補
models = {
    "knn": KNeighborsClassifier(n_jobs=2), # 時間かかる
    "dt": DecisionTreeClassifier(), # 時間かかる
    "rf": RandomForestClassifier(n_jobs=-1), # すぐ終わる
    "svm": SVC() # 時間かかる
}

# パーサーの作成
parser = argparse.ArgumentParser(
    description="K-NN, DT, RF or SVM trainer",
    add_help=True, # -h/–help オプションの追加
)

# 引数の追加
parser.add_argument(
    "model",
    help="select a model",
    choices=["knn", "dt", "rf", "svm"]
)

# 引数を解析し，モデル名を取得
args = parser.parse_args()
model_name = args.model


def run(model_name):
    """
    メインメソッド
    """

    # データの準備
    koma = LoadData()
    koma.run() # メインメソッドを実行して各プロパティにデータを格納する．
    target_names = koma.target_names # 駒の種類．混同行列に利用．
    x = koma.data.reshape(koma.data.shape[0], -1) # 一次元化
    y = koma.target_ids
    x_train, x_test, y_train, y_test = train_test_split(x, y,
        test_size=0.3, random_state=42)

    # モデルの選択
    model = models[model_name]

    # 学習済みモデルがあれば利用し，なければ学習させる
    learned_model_exists=True
    try:
        classifier = joblib.load("../result/{}.pkl".format(model_name))
    except FileNotFoundError:
        classifier = model.fit(x_train, y_train)
        model_exists=False

    # 予測
    y_pred = classifier.predict(x_test)

    # 結果の表示
    print(model.__class__.__name__)
    print("train:", classifier.score(x_train, y_train))
    print("test:", classifier.score(x_test, y_test))
    print("F1: ", f1_score(y_test[:len(y_pred)], y_pred, average='macro'))

    ## 正規化前の混合行列の可視化
    plot_confusion_matrix(y_test, y_pred, classes=target_names,
        title='Confusion matrix, without normalization')

    ##  正規化後の混合行列の可視化
    plot_confusion_matrix(y_test, y_pred, classes=target_names,
        normalize=True, title='Normalized confusion matrix')

    # 学習済みモデルがなかった場合はモデルを保存する
    if not learned_model_exists:
        joblib.dump(classifier, "../result/{}.pkl".format(model_name))


if __name__ == "__main__":
    run(model_name)
