import numpy as np
import time
from tqdm import tqdm

from sklearn.utils import shuffle

import chainer


def test(model, x_data, y_data, batchsize=10):
    N = x_data.shape[0]
    x_data, y_data = shuffle(x_data, y_data) # テストする順番をランダムに入れ替え

    sum_accuracy = 0    # 累計正答率
    sum_loss = 0        # 累計誤差

    # batchsize個ずつ評価
    for i in tqdm(range(0, N, batchsize)):
        # 評価の時はvolatile
        x = chainer.Variable(np.asarray(x_data[i: i+batchsize]))
        t = chainer.Variable(np.asarray(y_data[i: i+batchsize]))

        # 評価
        loss = model(x, t)
        sum_loss += float(loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)

    print("test mean loss={}, accuracy={}".format(sum_loss / N, sum_accuracy / N))