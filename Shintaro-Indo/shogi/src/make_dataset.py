import glob
import os
import os.path
import pickle
from zipfile import ZipFile

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class LoadData():
    """
    load_data.data：画像
    load_data.target：ラベル
    load_data.target_names：クラス名
    注1：上記3つは全てarray
    注2：パスは全てmake_datasetモジュールを読み込むファイルを起点とする
    """

    def __init__(self):
        self.data = []  # 画像を格納するlist．後にarrayに変換．
        self.target = []  # ラベルを格納するlist．後にarrayに変換．
        self.target_names = np.array([  # 成り駒以外の8種類
            "fu", "gin", "hisya", "kaku", "kei", "kin", "kyo", "ou"])
        self.run()

    def extract_zip(self, zip_dir_path, file_name): #
        """
        zipファイルを， zipファイルが存在するディレクトリで展開するメソッド
        input：zip_dir_path zipファイルが存在するディレクトリへのパス
        input：file_name zipファイルの名前
        """
        with ZipFile(zip_dir_path + file_name, "r") as z:
            z.extractall(zip_dir_path)

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
        data_dir_path = "../dataset/image/annotation_koma_merge/"

        # 各クラスごとに， 画像をself.dataに、ラベルをself.targetに格納する。
        for target, target_name in enumerate(self.target_names):

            # 画像へのパスを作成
            data_paths = glob.glob(data_dir_path + target_name + "/*")

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
        if (os.path.exists("../dataset/pickle.zip")
                and (os.path.exists("../dataset/pickle"))):
            self.extract_zip(data_dir_path="../dataset/",
                             file_name="pickle.zip")

        # pickleファイルがあればそこから読み込む
        elif os.path.exists("../dataset/pickle"):
            self.data = self.load_pickle(path="../dataset/pickle/data.pkl")
            self.target = self.load_pickle(path="../dataset/pickle/target.pkl")

        # 生データのzipしかなければ解凍する
        elif (os.path.exists("../dataset/image/annotation_koma_merge.zip")
                and os.path.exists("../dataset/image/annotation_koma_merge")):
            self.extract_zip(data_dir_path="../dataset/image/",
                             file_name="annotation_koma_merge.zip")

        # 生データからデータセットを作成し， pickle化する
        elif os.path.exists("../dataset/image/annotation_koma_merge"):

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
