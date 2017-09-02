** ツリー構造

  self-study/

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
