# 檔案用途:
./configs/                            存放設定\
./configs/configs.ini                 存放存儲目錄、版本號等\
./configs/CatBoostClassifier.json     存放catboost超參數\
./data/                               存放所有資料集\
./outputs/                            存放所有輸出\
./outputs/checkpoint/                 存放保存的模型\
./outputs/prediction/                 存放inference結果\
./outputs/logging/                    存放logger輸出\
data.py                               資料讀取和前處理\
model.py                              模型定義\
test_catboost.py                      訓練catboost\
tools.py                              一些工具function\
train_catboost.py                     訓練optuna\
trainer.py                            訓練和inference程式

# 執行流程:
## 檔案前處理
刪除data preprocess、inference、model、api這幾個檔案
執行目錄轉到code檔案下

## 安裝所需套件
$ pip install -r requirements.txt 

## 執行訓練optuna
$ python train_catboost.py

在./configs/configs.ini所指定的檔案中選擇超參數設定，輸入到./configs/CatBoostClassifier.json中

## 訓練catboost
$ python test_catboost.py
