import chainer
from chainer import Function, Variable, optimizers, Link, Chain
import chainer.functions as F
import chainer.links as L

class MLP(Chain):
    def __init__(self, n_units):
        super(MLP, self).__init__(
            l1 = L.Linear(None, n_units),
            l2 = L.Linear(n_units, n_units),
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        return self.l2(h1)