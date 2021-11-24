## データセットの用意


### 広告データセット


#### 1. imageのダウンロード

ソース:[Image Dataset](https://people.cs.pitt.edu/~kovashka/ads/#video)

```
Our image dataset contains a total of 64,832 advertisement images verified by human annotators on Amazon Mechanical Turk. Images are split into 11 folders (subfolder-0 to subfolder-10).
To obtain the dataset, please just download all compressed zip files and extract them all into the same folder.
```
以下の10個のzipをダウンロードしたのち、data/ads/imageフォルダの中に解凍する。つまり、data/ads/image/0/xxx.pngとなるように配置する。

* https://storage.googleapis.com/ads-dataset/subfolder-0.zip
* https://storage.googleapis.com/ads-dataset/subfolder-1.zip
* https://storage.googleapis.com/ads-dataset/subfolder-2.zip
* https://storage.googleapis.com/ads-dataset/subfolder-3.zip
* https://storage.googleapis.com/ads-dataset/subfolder-4.zip
* https://storage.googleapis.com/ads-dataset/subfolder-5.zip
* https://storage.googleapis.com/ads-dataset/subfolder-6.zip
* https://storage.googleapis.com/ads-dataset/subfolder-7.zip
* https://storage.googleapis.com/ads-dataset/subfolder-8.zip
* https://storage.googleapis.com/ads-dataset/subfolder-9.zip
* https://storage.googleapis.com/ads-dataset/subfolder-10.zip

#### 2.annotationのダウンロード（事前実行済みなので実行不要）

以下のリンクから画像セットのannotationのzipをダウンロードし、data/adsフォルダの中に解凍する。
解凍すると、annotationというフォルダがdata/ads/annotationという配置で置かれる。

* https://people.cs.pitt.edu/~kovashka/ads/annotations_images.zip


#### 3.image_listの作成（事前実行済みなので実行不要）

```
cd data/ads
python3 make_image_list.py
```

上記を実行することで、data/ads/image_name_listに、全画像のタイトルがリストになったテキストファイルが生成される。


#### 4. annotaitonから画像ごとの特徴量抽出

```
cd data/ads
python3 make_sentence_vectors.py
```

上記を実行すると、data/ads/image/0/xxx.txtのようにxxx.pngに対応したtxtファイルが作成される。このxxx.txtファイルにはdata/ads/annotation/QA_Action.json, data/ads/annotation/QA_Reason.jsonからxxx.pngに対応しているテキストを抽出し、 ベクトルに変換したものが保存されている。
また、ベクトルの変換にはBERTの学習済みモデルを使用した。


### 車データセット

#### 1. imageのダウンロード

ソース：[DVM-CAR データセット](https://deepvisualmarketing.github.io/)

上記からダウンロードしたzipをdata/carsフォルダに解凍すると、confirmed_frontsというフォルダができ、中には車種や年代で分類された車のフロント画像が入っている。つまり。data/cars/confirmed_frontsのように配置する。

#### 2.ファインチューニング
```
cd data/cars
python3 finetune.py
```
上記を実行すると、data/cars/color/colorhexa_com.csvに色名とRGB値がセットになったデータがあるので、これを使いBERTの学習済みモデルに対してファインチューニングを行う。
ファインチューニングされたモデルはdata/cars/transformersに保存される。
色のデータセットについては以下を参照してください。

https://data.world/dilumr/color-names

#### 3. annotationから画像ごとの特徴量抽出
```
cd data/cars
python3 make_sentence_vectors.py
```
上記を実行すると、data/ads/confirmed_fronts内にあるxxx.pngに対応したxxx.txtファイルが作成され、xxx.pngと同じディレクトリに配置される。このxxx.txtファイルにはxxx.pngの車の色名を抽出し、 ベクトルに変換したものが保存されている。

----

## 実験1: 広告画像のランダム生成



```=python
cd /src/EXP1_random_ads
python3 GAN_random_gen.py
```
上記を実行すろと、result/EXP1_random_ads/image内にランダム生成された画像が学習が経過する一定間隔で出力される。

*result/EXP1_random_ads/image_preには、こちらで実行した結果が置かれている。*


---

## 実験2: テキストから広告画像を生成する



### 学習の実行/モデルの生成

```=python
cd /src/EXP2_ads_text2image
python3 
```
これによって、result/EXP2_ads_text2image/imageに学習過程で生成された画像が、result/EXP2_ads_text2image/modelにモデルが出力される。


### モデルを使用して、テキストから画像を生成

```=python
cd /src/EXP2_ads_text2image
python3 
```
3単語の入力を求められる。必ず3単語の英語を入力する。
テキストから生成された画像はresult/EXP2_ads_text2image/outputに置かれる。

*result/EXP2_ads_text2image/output_preには、こちらで実行した結果が置かれている。*

---

## 実験3:　テキストから車画像を生成する



### 学習の実行/モデルの生成

```=python
cd /src/EXP3_cars_text2image
python3 GAN_text2image_cars.py
```
これによって、result/EXP3_cars_text2image/imageに学習過程で生成さらた画像が、result/EXP3_cars_text2image/modelにモデルが出力される。



### モデルを使用して、テキストから画像を生成

```=python
cd /src/EXP3_cars_text2image
python3 GAN_text2image_test_cars.py
```

3単語の入力を求められる。必ず3単語の英語を入力する。例えば、"black blue red" や、 "sky apple chocolate"など。
テキストから生成された画像はresult/EXP3_cars_text2image/outputに置かれる。

*result/EXP2_ads_text2image/output_preには、こちらで実行した結果が置かれている。*