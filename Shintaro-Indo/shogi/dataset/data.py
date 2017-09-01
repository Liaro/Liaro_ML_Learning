import numpy as np
import pickle
import glob
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from PIL import Image


class fetch_data():
    def __init__(self):
        self.data = [] # 画像を格納するlist．後にarrayに変換．
        self.target = [] # ラベルを格納するlist．後にarrayに変換．
        self.target_names = np.array(["fu", "gin", "hisya", "kaku", "kei", "kin", "kyo", "ou"]) # 成り駒以外の8種類
        self.make_dataset()


    def make_dataset(self):    
        # pickleファイルがあればそこから読み込む
        if ("../dataset/pickles/data.pickle" in glob.glob("../dataset/pickles/*")) and ("../dataset/pickles/target.pickle" in glob.glob("../dataset/pickles/*")):
            with open("../dataset/pickles/data.pickle", "rb") as f:
                self.data = pickle.load(f)

            with open("../dataset/pickles/target.pickle", "rb") as f:
                self.target = pickle.load(f)

        # pickleファイルがなければ生のデータを読み込みpickle化も行う。
        else:
            size = (64, 80) # 画像サイズ = (横, 縦)
            data_dir = "../dataset/images/annotation_koma_merge/" # 画像があるディレクトリ

            # 画像をself.dataに、ラベルをself.targetに格納する。
            for target, target_name in enumerate(self.target_names):
                data_paths = glob.glob(data_dir + target_name + "/*") # 画像へのパスを作成

                # 格納
                for data_path in data_paths:
                    self.data.append(np.array(Image.open(data_path).resize(size))[:, :, :3]) # 4channel目は無視．
                    self.target.append(target)

            # Arrayに変換
            self.data = np.array(self.data)
            self.target = np.array(self.target)

            # pickle化
            with open("../dataset/pickles/data.pickle", "wb") as f:
                pickle.dump(self.data, f)

            with open("../dataset/pickles/target.pickle", "wb") as f:
                pickle.dump(self.target, f)

