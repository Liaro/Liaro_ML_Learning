import glob
import os
import pickle
from zipfile import ZipFile

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class load_data():
    """
    load_data.data：画像
    load_data.target：ラベル
    load_data.target_names：クラス名
    注1：上記3つは全てarray．
    注2：パスは全てdatasetを読み込むファイルを起点とする．
    """

    def __init__(self):
        self.data = []  # 画像を格納するlist．後にarrayに変換．
        self.target = []  # ラベルを格納するlist．後にarrayに変換．
        self.target_names = np.array([  # 成り駒以外の8種類
            "fu", "gin", "hisya", "kaku", "kei", "kin", "kyo", "ou"])
        self.run()

    def extract_zip(self, dir_path, file_name): #
        """
        zipファイルを， zipファイルが存在するディレクトリで展開するメソッド
        input：dir_path zipファイルが存在するディレクトリへのパス
        input：file_name zipファイルの名前
        """
        with ZipFile(dir_path + file_name, "r") as z:
            z.extractall(dir_path)

    def load_pickle(self, path):
        """
        pickleファイルのデータを読み込んで，arrayを返すメソッド
        path：読み込むファイルからpickleファイルへのパス
        """
        with open(path, "rb") as f:
            return  pickle.load(f)

    def dump_pickle(self, path, data):
        """
        pickle化するメソッド
        path：作成するpickleファイルへのパス， data：pickle化するデータ
        """
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def make_dataset(self, size=(64, 80)):
        """
        生データからデータセットを作るメソッド(trainとtestの分け方はランダム)
        """

        # 生データが存在するディレクトリへのパス
        dir_path = "../dataset/image/annotation_koma_merge/"

        # 各クラスごとに， 画像をself.dataに、ラベルをself.targetに格納する。
        for target, target_name in enumerate(self.target_names):

            # 画像へのパスを作成
            data_paths = glob.glob(dir_path + target_name + "/*")

            # 格納
            for data_path in data_paths:
                self.data.append(np.array( # 4channel目は無視．
                    Image.open(data_path).resize(size))[:, :, :3])
                self.target.append(target)

        # Arrayに変換
        self.data = np.array(self.data)
        self.target = np.array(self.target)

    def run(self):
        """
        データセットに存在するデータの種類に応じて格納を行うメインメソッド
        """
        # pickleのzipしかなければ解凍する
        if (("../dataset/pickle.zip" in glob.glob("../dataset/*"))
                and ("../dataset/pickle" not in glob.glob("../dataset/*"))):
            self.extract_zip(dir_path="../dataset/", file_name="pickle.zip")

        # pickleファイルがあればそこから読み込む
        elif "../dataset/pickle" in glob.glob("../dataset/*"):
            self.data = self.load_pickle(path="../dataset/pickle/data.pkl")
            self.target = self.load_pickle(path="../dataset/pickle/target.pkl")

        # 生データのzipしかなければ解凍する
        elif (("../dataset/image/annotation_koma_merge.zip"
                in glob.glob("../dataset/image/*"))
                and("../dataset/image/annotation_koma_merge"
                not in glob.glob("../dataset/image/*"))):
            self.extract_zip(dir_path="../dataset/image/",
                             file_name="annotation_koma_merge.zip")

        # 生データからデータセットを作成し， pickle化する
        elif ("../dataset/image/annotation_koma_merge"
                in glob.glob("../dataset/image/*")):

            # データセットを作成
            self.make_dataset()

            # pickle化
            os.mkdir(path="../dataset/pickle")
            self.dump_pickle(path="../dataset/pickle/data.pkl",
                             data=self.data) # 画像データpickle化
            self.dump_pickle(path="../dataset/pickle/target.pkl",
                             data=self.target) # ラベルデータをpickle化

        # データがない場合はエラーメッセージを出力
        else:
            print("You have no available dataset")
