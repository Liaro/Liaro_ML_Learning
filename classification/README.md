# 分類
機械学習の中でも分類のタスクについて体験するスクリプト群

### 取り組み方
はじめに以下のnotebookを使って問題に取り組むために用いるツールの使い方を学ぶと良い  
how_to_svm_and_plot_CM.ipynb には、scikit-learnのSVMの使い方と分類問題の結果の可視化方法であるconfusion matrixの使い方をまとめている  
how_to_use_mecab.ipynb には、日本語の形態素解析ツールであるmecabのpythonバインディングの使い方をまとめている  
  
ツールの使い方を学んだあとは以下のnotebookを使って実際にニュース記事の分類問題を体験してみる
category_classification_svm.ipynb では、SVMを用いて日本語ニュース記事のカテゴリ分類とその評価を行う一連の流れを体験できる  
tfidf.ipynb では、上記に追加して、記事の特徴量の作成方法であるtfidfを用いた分類を体験できる

### 事前準備
分類問題を体験するnotebookでは以下のデータの分類する。  
取り組む前に、事前にダウンロード・解凍しておき、notebookと同じディレクトリの"data/classification/"以下に配置しておくこと。  

[livedoor ニュースコーパス](http://www.rondhuit.com/download.html#ldcc)

### 参考資料
[PythonでBag of WordsとSVMを使ったタイトルのカテゴリ分類](http://stmind.hatenablog.com/entry/2013/11/04/164608)  
[TF-IDFで文書内の単語の重み付け](https://takuti.me/note/tf-idf/)  
[Feature extraction(scikit-learn document)](http://scikit-learn.org/stable/modules/feature_extraction.html)