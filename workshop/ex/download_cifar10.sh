#! /bin/bash

# このリポジトリから拝借: https://github.com/mitmul/chainer-cifar10

wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar zxvf cifar-10-python.tar.gz
rm -rf cifar-10-python.tar.gz
python dataset.py
