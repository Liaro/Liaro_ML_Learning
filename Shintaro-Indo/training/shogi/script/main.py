from load_koma import load_koma
from train import train
from test import test
from mlp import MLP
from cnn import CNN

import numpy as np

from sklearn.model_selection import train_test_split

import chainer
from chainer import optimizers
import chainer.links as L


# データの読み込み
koma = load_koma()
x = koma.data
y = koma.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


# Chainerでは実数のタイプはnp.float32, 整数のタイプはnp.int32に固定しておく必要がある．
x_train = x_train.astype(np.float32) # (40681, 80, 64, 3)
y_train = y_train.astype(np.int32) # (40681,)
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.int32)


# 輝度を揃える
x_train /= x_train.max()
x_test /= x_test.max()


models = [
    MLP(1000),
    CNN()
]

model = L.Classifier(models[1]) # モデルの生成
optimizer = optimizers.Adam() # 最適化アルゴリズムの選択
optimizer.setup(model) # アルゴリズムにモデルをフィット


# 学習とテスト
n_epoch = 10 # 学習回数(学習データを何周するか)
for epoch in range(1, n_epoch + 1):
    print("\nepoch", epoch)

    # 訓練
    train(model, optimizer, x_train, y_train, batchsize=100)

    # 評価
    test(model, x_test, y_test, batchsize=100)
