**ツリー構造**

	self-study/ 自習用
		
		┣ dataset/
			┣ mldata
		┣ notebook/ ← プロトタイプ  
			┣ load_data.pickle
			┣ nn.pickle  
			┣ non_nn.pickle  
		┣ src/  
			┣ data.py： pickleファイルがあれば読み込み， なければ生データからデータセットを作成してpickle化も行う
			┣ non_nn.py： NN以外の学習  
			┣ train.py： NNの学習  
			┣ cnn.py
			┣ mlp.py
			┣ resnet.py
		┣ result/



	shogi/  将棋の駒画像の分類
		┣ dataset/
			┣ images/ ← ignore
				┣ annotation_koma_merge/...  
			┣ pickles/ ← ignore
				┣ data.pickle
				┣ target.pickle  
		┣ notebook/ ← プロトタイプ  
			┣ load_data.ipynb
			┣ nn.ipynb
			┣ non_nn.ipynb
		┣ src/  
			┣ data.py： pickleファイルがあれば読み込み， なければ生データからデータセットを作成してpickle化も行う
			┣ non_nn.py： NN以外の学習  
			┣ train.py： NNの学習  
			┣ cnn.py
			┣ mlp.py
			┣ resnet.py
		┣ result/
