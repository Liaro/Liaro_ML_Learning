import numpy as np
import pickle
import glob
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from PIL import Image
from zipfile import ZipFile
import os

# load_data.data, load_data.target, load_data.target_namesで，それぞれ画像，ラベル，クラス名にアクセスできる．全てarray．
class load_data(): # 注：パスは全てload_dataを読み込むファイルを起点とする．
    def __init__(self):
        self.data = [] # 画像を格納するlist．後にarrayに変換．
        self.target = [] # ラベルを格納するlist．後にarrayに変換．
        self.target_names = np.array(["fu", "gin", "hisya", "kaku", "kei", "kin", "kyo", "ou"]) # 成り駒以外の8種類
        self.run() # データセットに存在するデータの種類に応じて格納を行うメインメソッド

    # zipファイルを， zipファイルが存在するディレクトリで展開するメソッド
    def extract_zip(self, dir_path, file_name): # dir_path：zipファイルが存在するディレクトリへのパス file_name：zipファイルの名前
        with ZipFile(dir_path + file_name, "r") as z:
            z.extractall(dir_path)

    # pickleファイルのデータを読み込んで，arrayを返すメソッド
    def load_pickle(self, path): # path：読み込むファイルからpickleファイルへのパス
        with open(path, "rb") as f:
            return  pickle.load(f)

    # pickle化するメソッド
    def dump_pickle(self, path, data): # path：作成するpickleファイルへのパス， data：pickle化するデータ
        with open(path, "wb") as f:
            pickle.dump(data, f)

    # 生データからデータセットを作るメソッド(trainとtestの分け方はランダム)
    def make_dataset(self, size=(64, 80)):
        # 生データが存在するディレクトリへのパス
        dir_path = "../dataset/image/annotation_koma_merge/"

        # 各クラスごとに， 画像をself.dataに、ラベルをself.targetに格納する。
        for target, target_name in enumerate(self.target_names):

            # 画像へのパスを作成
            data_paths = glob.glob(dir_path + target_name + "/*")

            # 格納
            for data_path in data_paths:
                self.data.append(np.array(Image.open(data_path).resize(size))[:, :, :3]) # 4channel目は無視．
                self.target.append(target)

        # Arrayに変換
        self.data = np.array(self.data)
        self.target = np.array(self.target)


    # データセットに存在するデータの種類に応じて格納を行うメインメソッド
    def run(self):
        # pickleのzipしかなければ解凍する
        if ("../dataset/pickle.zip" in glob.glob("../dataset/*")) and ("../dataset/pickle" not in glob.glob("../dataset/*")):
            self.extract_zip(dir_path="../dataset/", file_name="pickle.zip")

        # pickleファイルがあればそこから読み込む
        elif "../dataset/pickle" in glob.glob("../dataset/*"):
            self.data = self.load_pickle(path="../dataset/pickle/data.pkl")
            self.target = self.load_pickle(path="../dataset/pickle/target.pkl")

        # 生データのzipしかなければ解凍する
        elif ("../dataset/image/annotation_koma_merge.zip" in glob.glob("../dataset/image/*")) and ("../dataset/image/annotation_koma_merge" not in glob.glob("../dataset/image/*")):
            self.extract_zip(dir_path="../dataset/image/", file_name="annotation_koma_merge.zip")

        # 生データからデータセットを作成し， pickle化する
        elif "../dataset/image/annotation_koma_merge" in glob.glob("../dataset/image/*"):

            # データセットを作成
            self.make_dataset()

            # pickle化
            os.mkdir(path="../dataset/pickle")
            self.dump_pickle(path="../dataset/pickle/data.pkl", data=self.data) # 画像データpickle化
            self.dump_pickle(path="../dataset/pickle/target.pkl", data=self.target) # ラベルデータをpickle化

        # データがない場合はエラーメッセージを出力
        else:
            print("You have no available dataset")
