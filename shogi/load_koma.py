import numpy as np
import pickle
import glob
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from PIL import Image


class load_koma():
    def __init__(self):
        self.target_names = np.array(["fu", "gin", "hisya", "kaku", "kei", "kin", "kyo", "ou"]) # 成駒以外：8クラス

        # pickleファイルがあれば読み込む
        if ("./koma_data.pickle" in glob.glob("./*")) and (("./koma_target.pickle" in glob.glob("./*"))):
            with open("koma_data.pickle", "rb") as f:
                self.data = pickle.load(f)

            with open("koma_target.pickle", "rb") as f:
                self.target = pickle.load(f)

        # pickleファイルなければ作る。
        else:
            self.data = []
            self.target = []

            size = (64, 80) # 画像サイズ = (横, 縦)
            data_dir = "../data/annotation_koma_merge/" # 画像が存在するディレクトリ

            # 画像をself.dataに、ラベルをself.targetに格納する。
            for target, target_name in enumerate(self.target_names):
                # 画像へのパスを作成
                data_paths = glob.glob(data_dir + target_name + "/*")
                # 格納
                for data_path in data_paths:
                    self.data.append(np.array(Image.open(data_path).resize(size))[:, :, :3]) # 4channel目は無視．
                    self.target.append(target)

            # Arrayに変換
            self.data = np.array(self.data)
            self.target = np.array(self.target)

            # pickle化
            with open("koma_data.pickle", "wb") as f:
                pickle.dump(self.data, f)

            with open("koma_target.pickle", "wb") as f:
                pickle.dump(self.target, f)
