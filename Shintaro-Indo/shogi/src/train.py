import sys,os
sys.path.append(os.pardir)

from load_data import load_data
from mlp import MLP
from cnn import CNN
from resnet import ResNetSmall, ResBlock

import time
from tqdm import tqdm
import numpy as np

from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import chainer
from chainer import cuda
from chainer import optimizers
import chainer.links as L


# モデルの候補
models = {
    "mlp": MLP(1000),
    "cnn": CNN(),
    "resnet": ResNetSmall()
}


# GPUが使える場合はGPU対応に
gpu = False
try:
    gpu_device = 0
    cuda.get_device(gpu_device).use()
    xp = cuda.cupy
    gpu = True
except:
    xp = np

# 前処理
def preprocessing():
    # データの読み込み
    koma = load_data()
    x = koma.data
    x = x.reshape(x.shape[0], 3, 80, 64) # (データ数，チャネル数(色数)，縦，横)の形式にする．
    y = koma.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Chainerでは実数のタイプはfloat32, 整数のタイプはint32に固定しておく必要がある．
    x_train = x_train.astype(xp.float32) # (40681, 80, 64, 3)
    y_train = y_train.astype(xp.int32) # (40681,)
    x_test = x_test.astype(xp.float32)
    y_test = y_test.astype(xp.int32)

    # 輝度を揃える
    x_train /= x_train.max()
    x_test /= x_test.max()
    return x_train, y_train, x_test, y_test


# 訓練データに対する正答率，誤差を表示する関数
def train(model, optimizer, x_data, y_data, batchsize=10):
    N = x_data.shape[0] # データ数
    x_data, y_data = shuffle(x_data, y_data) # 学習する順番をランダムに入れ替え


    sum_accuracy = 0 # 累計正答率
    sum_loss = 0 # 累計誤差
    start = time.time() # 開始時刻

    # batchsize個ずつ学習
    for i in tqdm(range(0, N, batchsize)):
        x = chainer.Variable(xp.asarray(x_data[i: i+batchsize]))
        t = chainer.Variable(xp.asarray(y_data[i: i+batchsize]))

        # パラメータの更新(学習)
        optimizer.update(model, x, t)

        sum_loss += float(model.loss.data) * len(t.data) # 累計誤差を更新
        sum_accuracy += float(model.accuracy.data) * len(t.data) # 累計正答率を更新

    end = time.time() # 終了時刻
    elapsed_time = end - start # 所要時間
    throughput = N / elapsed_time # 単位時間当たりの作業量
    print("train mean loss={}, accuracy={}, throughput={} image/sec".format(sum_loss / N, sum_accuracy / N, throughput))


# テストデータに対する正答率，誤差を表示する関数
def test(model, x_data, y_data, batchsize=10):
    N = x_data.shape[0]
    x_data, y_data = shuffle(x_data, y_data)

    sum_accuracy = 0
    sum_loss = 0

    for i in tqdm(range(0, N, batchsize)):
        # 評価の時はvolatile *volatileをTrueにするとBackpropergationできない
        x = chainer.Variable(xp.asarray(x_data[i: i+batchsize]))
        t = chainer.Variable(xp.asarray(y_data[i: i+batchsize]))

        # 評価
        loss = model(x, t)
        sum_loss += float(loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)

    print("test mean loss={}, accuracy={}".format(sum_loss / N, sum_accuracy / N))


if __name__ == "__main__":
    model_name = sys.argv[1]
    if len(sys.argv) == 2 and model_name in models.keys(): # コマンドライン引数が条件を満たしているとき
        # データの準備
        x_train, y_train, x_test, y_test = preprocessing()

        # モデルと最適化アルゴリズムの設定
        model = L.Classifier(models[model_name]) # モデルの生成

        # 学習済みモデルが存在する場合は利用する
        try:
            serializers.load_npz("{}.npz".format(model_name), model)
        except:
            pass

        # GPUが使える場合はGPU対応に，
        if gpu:
            model = model.to_gpu(gpu_device)

        optimizer = optimizers.Adam() # 最適化アルゴリズムの選択
        optimizer.setup(model) # アルゴリズムにモデルをフィット

        # 学習
        n_epoch = 10 # 学習回数(学習データを何周するか)
        for epoch in range(1, n_epoch + 1):
            print("\nepoch", epoch)

            # 訓練
            train(model, optimizer, x_train, y_train, batchsize=100)

            # 評価
            test(model, x_test, y_test, batchsize=100)

        # モデルの保存
        model.to_cpu() # CPUで計算できるようにしておく
        serializers.save_npz("{}.npz".format(model_name), model) # npz形式で書き出し

    else: # 例外処理
        print("please specify the model (mlp, cnn or resnet) like $ python train.py cnn")
