# 将棋の駒画像の分類

**ツリー構造**

	shogi/
		┣ dataset/ ← ignore
			┣ image/
				┣ annotation_koma_merge/
					┣ fu
					┣ gin ...etc
			┣ pickle/
				┣ data.pickle
				┣ target.pickle  
		┣ notebook/ ← プロトタイプ  
			┣ load_data.ipynb
			┣ nn.ipynb
			┣ non_nn.ipynb
		┣ src/  
			┣ load_data.py： データがある場合，インスタンス変数 data, target, target_namesで画像，ラベル，クラス名にアクセスできるようにする．
			┣ non_nn.py： NNを利用しないモデルの学習  
			┣ train.py： NNを利用したモデルの学習  
			┣ cnn.py
			┣ mlp.py
			┣ resnet.py
		┣ result/ 学習済みモデル
			┣ rf.pkl


**結果**

	RF
	- 前処理なし：(train, test, F1) = (0.9997, 0.9859, 0.9840)
	- 適当な閾値(定数)で二値化：(train, test, F1) = (0.8273, 0.7779, 0.6552)  
	　∵ 画像によっては真っ黒(白)になってしまう
	　* kNN, DT, SVMはCPUだと時間ががかるので保留中

	MLP
	- (train, test, F1) = (0.9035, 0.8631, - )  
	  * チューニングは未

	CNN
	- (train, test, F1) = (0.9916, 0.9928, - )    

	ResNet
	- (train, test, F1) = (0.9834, 0.97765, - )
