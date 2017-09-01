#(入力チャネル数，出力フィルタ数，カーネルサイズ，ストライド(1)，パディング(0)）
# xを足すので，入力サイズと揃える必要あり

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import initializers

class ResBlock(chainer.Chain):

    def __init__(self, in_size, ch):
        super(ResBlock, self).__init__()
        initialW = initializers.HeNormal(),
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_size, ch, 1, 1, 0, initialW=initialW, nobias=True),
            self.bn1 = L.BatchNormalization(ch),
            self.conv2 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=initialW, nobias=True),
            self.bn2 = L.BatchNormalization(ch),
            self.conv3 = L.Convolution2D(ch, in_size, 1, 1, 0, initialW=initialW, nobias=True),
            self.bn3 = L.BatchNormalization(in_size),


    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))

        return F.relu(h + x)

class ResNetSmall(chainer.Chain):

    def __init__(self):
        super(ResNetSmall, self).__init__(
            conv1 = L.Convolution2D(3, 32, 5, initialW=initializers.HeNormal()),
            bn1 = L.BatchNormalization(32),
            res1= ResBlock(32, 32),
            res2= ResBlock(32, 32),
            res3= ResBlock(32, 32),
            fc=L.Linear(None, 10),
        )

    def __call__(self, x):
        h = self.bn1(self.conv1(x))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = self.res2(h)
        h = self.res3(h)
        h = F.average_pooling_2d(h, 7, stride=1)
        return self.fc(h)
