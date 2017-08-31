from load_koma import load_koma

import numpy as np
import cv2
import itertools
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.tree import  DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# kNN, DT, RF, SVMで認識． 
class NonNN:
    def __init__(self, x_train, y_train, x_test, y_test, class_names, threshold):
        self.x_train = x_train.reshape(x_train.shape[0], -1) # 一次元化
        self.y_train = y_train
        self.x_test = x_test.reshape(x_test.shape[0], -1)
        self.y_test = y_test
        self.class_names = class_names # 駒の名前(混同行列で利用)
        self.threshold = threshold # 閾値処理
        self.models = [
            KNeighborsClassifier(n_jobs=2), 
            DecisionTreeClassifier(), 
            RandomForestClassifier(n_jobs=-1), 
            SVC()
        ]
              
            
    # スコアを表示
    def show_score(self, model_index=2):
        # グレースケール化 → 閾値処理
        if self.threshold:
            # グレースケール化
            self.x_train =  np.array([cv2.cvtColor(train_img, cv2.COLOR_RGB2GRAY) for train_img in self.x_train])
            self.x_test = np.array([cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY) for test_img in self.x_test])
            # 閾値処理
            self.x_train = np.array([cv2.threshold(train_img,  170, 255, cv2.THRESH_BINARY)[1] for train_img in self.x_train])
            self.x_test  = np.array([cv2.threshold(test_img,  170, 255, cv2.THRESH_BINARY)[1] for test_img in self.x_test])
                           
        model = self.models[model_index]
        clf = model.fit(self.x_train, self.y_train)
        self.y_pred = clf.predict(self.x_test)
        print(model.__class__.__name__)
        print("train:", clf.score(self.x_train, self.y_train))
        print("test:", clf.score(self.x_test, self.y_test))
        print("F1: ", f1_score(self.y_test[:len(self.y_pred)], self.y_pred, average='macro'))  
    
    
     # 混同行列を描画
    def plot_confusion_matrix(self, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        #  混同行列の作成
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        # 正規化
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # 行の和を列ベクトル化

        plt.figure(figsize = (12, 12))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45) 
        plt.yticks(tick_marks, classes) 

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label') 
        plt.xlabel('Predicted label')
        
        
    # スコアと混同行列を表示するためのメソッド
    def show_result(self):
        # スコアの表示
        self.show_score(model_index=2)
        
        # 正規化前の混合行列の可視化
        # plt.figure()
        self.plot_confusion_matrix(classes=self.class_names, title='Confusion matrix, without normalization')

        # 正規化後の混合行列の可視化
        # plt.figure()
        self.plot_confusion_matrix(classes=self.class_names, normalize=True, title='Normalized confusion matrix')

        plt.show()
        

# データの読み込み
koma = load_koma()
x = koma.data
y = koma.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# 駒の種類
class_names = ["fu", "gin", "hisya", "kaku", "kei", "kin", "kyo", "ou"]

non_nn = NonNN(x_train, y_train, x_test, y_test, class_names, threshold=False)
non_nn.show_result()

