import numpy as np
import time
from tqdm import tqdm

from sklearn.utils import shuffle

import chainer


def train(model, optimizer, x_data, y_data, batchsize=10):
    N = x_data.shape[0]
    x_data, y_data = shuffle(x_data, y_data) # 学習する順番をランダムに入れ替え

    sum_accuracy = 0    # 累計正答率
    sum_loss = 0        # 累計誤差
    start = time.time() # 開始時刻

    # batchsize個ずつ学習
    for i in tqdm(range(0, N, batchsize)):
        x = chainer.Variable(np.asarray(x_data[i: i+batchsize]))
        t = chainer.Variable(np.asarray(y_data[i: i+batchsize]))

        # パラメータの更新(学習)
        optimizer.update(model, x, t)

        sum_loss += float(model.loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)

    end = time.time() # 終了時刻
    elapsed_time = end - start
    throughput = N / elapsed_time # 単位時間当たりの作業量
    print("train mean loss={}, accuracy={}, throughput={} image/sec".format(sum_loss / N, sum_accuracy / N, throughput))