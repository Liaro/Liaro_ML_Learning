# 将棋の駒画像の分類

**ツリー構造**

	shogi/  
		┣ dataset/
			┣ data.py： pickleファイルがあれば読み込み， なければ生データからデータセットを作成してpickle化も行う
			┣ image/ ← 重いためあげていない
				┣ annotation_koma_merge/  
			┣ pickles/ ← 重いためあげていない
				┣ data.pickle
				┣ target.pickle  
		┣ result/ ← 未
		┣ src/   
			┣ cnn.py  
			┣ mlp.py  
			┣ non_nn.py： NN以外の学習  
			┣ train.py： NNの学習  
	　　

**結果**

	RF
	- 前処理なし：(train, test, F1) = (0.9997, 0.9859, 0.9840)
	- 適当な閾値(定数)で二値化：(train, test, F1) = (0.8273, 0.7779, 0.6552)  
	　∵ 画像によっては真っ黒(白)になってしまう
	　* kNNN，SVMはCPUだと時間ががかるので保留中

	MLP
	- (train, test, F1) = (0.9953, 0.9742, - )  
	  * チューニングは未
		  
	CNN
	- 未


