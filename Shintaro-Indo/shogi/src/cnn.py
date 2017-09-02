import chainer
from chainer import Function, Variable, optimizers, Link, Chain
import chainer.functions as F
import chainer.links as L

class CNN(chainer.Chain):
    def __init__(self):
        super(CNN, self).__init__(
            conv1 = L.Convolution2D(3, 32, 3, stride=1, pad=2), # (入力チャンネル数、出力チャンネル数、フィルタサイズ)
            conv2 = L.Convolution2D(32, 32, 3, stride=1, pad=2),
            conv3 = L.Convolution2D(32, 64, 3, stride=1, pad=2),
            fc4 = L.Linear(None, 64*10*8),
            fc5 = L.Linear(None, 8),
        )

    def __call__(self, x):
        h = F.relu(self.conv1(x)) # (32, 80, 64)
        h = F.max_pooling_2d(h, 2) # (32, 40, 32)
        h = F.relu(self.conv2(h))  # (32, 40, 32)
        h = F.max_pooling_2d(h, 2) # (32, 20, 16)
        h = F.relu(self.conv3(h)) # (64, 20, 16)
        h = F.dropout(F.relu(self.fc4(h)), ratio=0.5) # (64, 10, 8)
        return self.fc5(h)
